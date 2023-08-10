import torch
import torch.nn as nn

import abc
import numpy as np

from ..utils.prune_ops import *


class BaseNode(abc.ABC):
    def __init__(self, name: str, module: (None, nn.Module), grad) -> None:
        self.name = name
        self.module = module
        self.grad = grad
        self.round_to = 1
        self.split = 1
        self.next = []
        self.next_key = []
        self.next_inin = []
        self.prev = []
        self.prev_key = []
        self.prev_inin = []
        self.key = False
        self.is_prunable = False
        self.group = None
        self.in_param = []
        self.out_param = []
        self.prune_idx = [None, None]
        self.saved_idx = [None, None]
        # norm case: prune dim for input/output feature map is 1
        # special case:
        #   1. permute change the prune dim
        #   2. reshape may change the prune dim
        self.dim_offset = None

    def update_dim_offset(self, dim_offset):
        if self.dim_offset is None:
            self.dim_offset = dim_offset
            self.update_next_dim_offset(self.dim_offset)

    def update_next_dim_offset(self, dim_offset):
        for node in self.next:
            node.update_dim_offset(dim_offset)

    @abc.abstractmethod
    def prune(self):
        pass

    @abc.abstractmethod
    def add_prune_idx(self, prune_idx, prune_dim) -> bool:
        pass

    def add_prune_idx_tonext(self, prune_idx):
        for node in self.next:
            node.add_prune_idx(prune_idx, IDX_IN)

    def get_prune_param(self):
        if self.in_param == []:
            self.in_param = self.get_in_param()
        if self.out_param == []:
            self.out_param = self.get_out_param()
        return self.in_param, self.out_param

    # @abc.abstractmethod
    def get_in_param(self):
        pass

    # @abc.abstractmethod
    def get_out_param(self):
        pass

    def add_group(self, group):
        self.group = group

    def add_next(self, next_node):
        self.next.append(next_node)
        next_node.add_prev(self)

    def add_prev(self, prev_node):
        self.prev.append(prev_node)

    def find_next_key(self):
        if self.next_key != []:
            return self.next_key

        for node in self.next:
            if node.key:
                self.next_key.append(node)
            else:
                self.next_key.extend(node.find_next_key())
        return self.next_key

    def find_prev_key(self):
        if self.prev_key != []:
            return self.prev_key
        for node in self.prev:
            if node.key:
                self.prev_key.append(node)
            else:
                self.prev_key.extend(node.find_prev_key())
        return self.prev_key

    def find_next_inin(self):
        if self.next_inin != []:
            return self.next_inin
        for node in self.next_key:
            if isinstance(node, InInNode):
                self.next_inin.append(node)
        return self.next_inin

    def find_prev_inin(self):
        if self.prev_inin != []:
            return self.prev_inin
        for node in self.prev_key:
            if isinstance(node, InInNode):
                self.prev_inin.append(node)
        return self.prev_inin

        self.in_shape = 0
        self.out_shape = 0

    def update_shape(self):
        self.in_shape = self.get_in_shape().copy()
        self.out_shape = self.get_out_shape().copy()

    def get_in_shape(self):
        if hasattr(self, "in_shape") and self.in_shape != 0:
            return self.in_shape
        else:
            self.in_shape = self._get_in_shape().copy()
            return self.in_shape

    def get_out_shape(self):
        if hasattr(self, "out_shape") and self.out_shape != 0:
            return self.out_shape
        else:
            self.out_shape = self._get_out_shape().copy()
            return self.out_shape


class InOutNode(BaseNode):
    def __init__(self, name: str, module: nn.Module, grad) -> None:
        super().__init__(name, module, grad)
        self.key = True
        self.is_prunable = True
        if hasattr(grad, "metadata"):
            self.in_shape = list(grad.metadata["input"].shape)
            self.out_shape = list(grad.metadata["output"].shape)

    def prune(self):
        self.saved_idx[IDX_IN] = get_saved_idx(
            self.prune_idx[IDX_IN], self.module.weight.shape[DIM_IN]
        )
        self.saved_idx[IDX_OUT] = get_saved_idx(
            self.prune_idx[IDX_OUT], self.module.weight.shape[DIM_OUT]
        )
        self.prune_fn(self.module, self.saved_idx[IDX_IN], DIM_IN)
        self.prune_fn(self.module, self.saved_idx[IDX_OUT], DIM_OUT)

    def add_prune_idx(self, prune_idx, prune_dim):
        self.prune_idx[prune_dim] = prune_idx
        if prune_dim == IDX_OUT:
            self.add_prune_idx_tonext(prune_idx)
        return True


class ConvNode(InOutNode):
    def __init__(self, name: str, module: nn.Module, grad) -> None:
        super().__init__(name, module, grad)
        self.prune_fn = prune_conv
        self.in_shape = list(grad._saved_input.shape)
        self.in_channels = module.in_channels
        self.out_channels = module.out_channels


class LinearNode(InOutNode):
    def __init__(self, name: str, module: nn.Module, grad) -> None:
        super().__init__(name, module, grad)
        self.prune_fn = prune_fc

        self.in_channels = module.in_features
        self.out_channels = module.out_features


class LastLinearNode(InOutNode):
    def __init__(self, name: str, module: nn.Module, grad) -> None:
        super().__init__(name, module, grad)
        self.prune_fn = prune_fc
        self.in_channels = self.in_shape[-1]
        self.out_channels = self.out_shape[-1]
        self.prune_idx[IDX_OUT] = []

    def add_prune_idx(self, prune_idx, prune_dim):
        if self.dim_offset == len(self.in_shape) - 2:
            self.prune_idx[prune_dim] = prune_idx
            if prune_dim == IDX_OUT:
                self.prune_idx[
                    prune_dim
                ] = []  # FIXME: does not pruning the last layer output
                self.add_prune_idx_tonext(prune_idx)
            return True
        else:
            self.prune_idx[prune_dim] = []
            if prune_dim == IDX_IN:
                self.add_prune_idx_tonext(prune_idx)
            # if prune_dim == IDX_OUT:
            #     self.prune_idx[prune_dim] = []
            return True


# DONE: change the in/out channels of BundleParamNode later
#       now, we need to fix the param to the other output's shape
#       eg. make [4, 4] to [1, 1, 4, 4] for conv's output
#       eg. make [4] to [1, 4] for linear's output


class BundleParamNode(InOutNode):
    def __init__(self, name: str, param) -> None:
        super().__init__(name, None, None)
        self.param = param
        self.in_shape = list(param.shape)
        self.out_shape = self.in_shape.copy()
        self.in_channels = self.in_shape[1]
        self.out_channels = self.in_shape[1]

    def prune(self):
        self.saved_idx[IDX_OUT] = get_saved_idx(
            self.prune_idx[IDX_OUT], self.out_channels
        )
        prune_bundle(self.param, self.saved_idx[IDX_OUT], IDX_OUT)

    def add_prune_idx(self, prune_idx, prune_dim):
        assert prune_dim == IDX_OUT, f"expected dim {IDX_OUT}, got {prune_dim}"
        if self.out_channels == 1:
            self.prune_idx[IDX_OUT] = []
        else:
            self.prune_idx[IDX_OUT] = prune_idx
        return True


class OutOutNode(BaseNode):
    def __init__(self, name: str, module: nn.Module, grad) -> None:
        super().__init__(name, module, grad)
        self.in_shape = list(grad._saved_input.shape)
        self.out_shape = self.in_shape.copy()
        self.is_prunable = True

    def prune(self):
        self.saved_idx[IDX_OUT] = get_saved_idx(
            self.prune_idx[IDX_OUT], self.module.weight.shape[DIM_OUT]
        )
        self.prune_fn(self.module, self.saved_idx[IDX_OUT], DIM_OUT)

    def add_prune_idx(self, prune_idx, prune_dim):
        assert prune_dim == IDX_IN, f"expected dim {IDX_IN}, got {prune_dim}"
        self.prune_idx[IDX_IN] = prune_idx
        self.prune_idx[IDX_OUT] = prune_idx
        self.add_prune_idx_tonext(prune_idx)
        return True


class BatchNormNode(OutOutNode):
    def __init__(self, name: str, module: nn.Module, grad) -> None:
        super().__init__(name, module, grad)
        self.prune_fn = prune_batchnorm
        self.in_channels = module.num_features
        self.out_channels = module.num_features


class LayerNormNode(OutOutNode):
    def __init__(self, name: str, module: nn.Module, grad) -> None:
        super().__init__(name, module, grad)
        self.prune_fn = prune_layernorm
        self.in_channels = module.normalized_shape[0]
        self.out_channels = module.normalized_shape[0]


class GroupNormNode(OutOutNode):
    def __init__(self, name: str, module: nn.Module, grad) -> None:
        super().__init__(name, module, grad)
        self.prune_fn = prune_groupnorm
        self.in_channels = module.num_channels
        self.out_channels = module.num_channels


class GroupConvNode(OutOutNode):
    def __init__(self, name: str, module: nn.Module, grad) -> None:
        super().__init__(name, module, grad)
        self.prune_fn = prune_groupconv
        self.in_channels = module.in_channels
        self.out_channels = module.out_channels


# class LastLinearNode(OutOutNode):
#     def __init__(self, name: str, module: nn.Module, grad) -> None:
#         super().__init__(name, module, grad)


# FIXME: in_channels and out_channels
#        This maybe cause bug, test this in later version
class InInNode(BaseNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, None, grad)
        self.key = True
        self.is_prunable = True
        self.pruned = False

    def prune(self):
        pass

    def _get_in_shape(self):
        self.out_shape = list(self.next[0].get_in_shape())
        self.in_shape = self.out_shape.copy()
        return self.in_shape

    def _get_out_shape(self):
        self.in_shape = list(self.prev[0].get_out_shape())
        self.out_shape = self.in_shape.copy()
        return self.out_shape

    def update_shape(self):
        super().update_shape()
        self.in_channels = self.in_shape[1]
        self.out_channels = self.out_shape[1]

    def add_prune_idx(self, prune_idx, prune_dim):
        if prune_dim == IDX_IN:
            return False
        if self.pruned:
            return True
        self.prune_idx[prune_dim] = prune_idx
        self.add_prune_idx_tonext(prune_idx)
        self.pruned = True
        return True


class AddNode(InInNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)


class SubNode(InInNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)


class MulNode(InInNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)


class DivNode(InInNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)


# FIXME: q@k.T error, need to update the output shape
class MatmulNode(InInNode):
    """
    In most cases, this node used in qkv calculation.
    As a results, we do not need to consider the dim
    """

    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)
        self.in_shape = list(grad._saved_self.shape)
        self.in_shape_other = list(grad._saved_mat2.shape)
        self.out_shape = self.in_shape.copy()
        self.out_shape[-1] = self.in_shape_other[-1]
        self.prev_count = 0

    # TODO: how to decide the next node's prune_idx
    def add_prev(self, prev_node):
        super().add_prev(prev_node)
        if self.prev_count == 0:
            prev_node.out_shape = self.in_shape.copy()
            self.prev_count += 1
        elif self.prev_count == 1:
            prev_node.out_shape = self.in_shape_other.copy()
            self.prev_count += 1
        else:
            raise ValueError("MatmulNode can only have two prev node")

    # def update_dim_offset(self, dim_offset):
    #     if self.dim_offset is None:
    #         self.dim_offset = dim_offset
    #         self.update_next_dim_offset(self.dim_offset)

    def add_prune_idx(self, prune_idx, prune_dim):
        if prune_dim == IDX_IN:
            return False
        if self.pruned:
            return True
        self.prune_idx[prune_dim] = prune_idx
        self.add_prune_idx_tonext(prune_idx)
        self.pruned = True
        return True


class RemapNode(BaseNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, None, grad)
        self.key = True
        self.in_channels = 0
        self.out_channels = 0

    def prune(self):
        pass

    @abc.abstractmethod
    def add_prune_idx(self, prune_idx, prune_dim):
        pass


class ConcatNode(RemapNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)
        self.dim = grad._saved_dim
        self.prev_order2key = {}
        self.prev_order_count = 0
        self.add_prune_idx_count = 0

    def _get_in_shape(self):
        dim_sum = 0
        for node in self.prev:
            in_shape = node.get_out_shape()
            dim_sum += in_shape[self.dim]
        self.in_shape = in_shape
        self.in_shape[self.dim] = dim_sum
        self.out_shape = self.in_shape.copy()
        self.out_channels = self.out_shape[1]
        self.in_channels = self.out_channels
        return self.in_shape

    def _get_out_shape(self):
        return self.get_in_shape()

    # DONE: when concat's prev count equal to self.ratio, then re-generate the prune_idx
    def add_prev(self, prev_node):
        self.prev.append(prev_node)
        self.prev_order2key[self.prev_order_count] = prev_node.name
        self.prev_order_count += 1

    # FIXME: check this function
    def add_prune_idx(self, prune_idx, prune_dim):
        if prune_dim == IDX_OUT:
            if self.prune_idx[IDX_OUT] != None:
                self.add_prune_idx_tonext(prune_idx)
            else:
                return False
        elif prune_dim == IDX_IN:
            self.add_prune_idx_count += 1
            if self.add_prune_idx_count == self.prev_order_count:
                self.regenerate_prune_idx()
                self.group.add_prune_idx(self.prune_idx[IDX_OUT])
                # self.add_prune_idx_tonext(self.prune_idx[IDX_OUT])
        return True

    def regenerate_prune_idx(self):
        tmp_prune_idx = []
        length_count = 0
        for i, n in enumerate(self.prev):
            tmp_prune_idx.append((torch.tensor(n.prune_idx[IDX_OUT]) + length_count))
            length_count += n.out_channels
        self.prune_idx[IDX_IN] = torch.cat(tmp_prune_idx).tolist()
        self.prune_idx[IDX_OUT] = self.prune_idx[IDX_IN]


class SplitNode(RemapNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)
        self.in_shape = list(grad._saved_self_sym_sizes)
        self.out_shape = self.in_shape.copy()
        self.dim = (
            grad._saved_dim
            if grad._saved_dim < len(self.out_shape)
            else grad._saved_dim - 18446744073709551616
        )
        self.out_shape[self.dim] = grad._saved_split_size
        self.next_order2key = {}
        self.next_order_count = 0
        self.round_to = self.in_shape[self.dim] // self.out_shape[self.dim]
        self.split = self.round_to
        self.in_channels = self.in_shape[1]
        self.out_channels = self.out_shape[1]

    def add_next(self, next_node):
        self.next.append(next_node)
        next_node.add_prev(self)
        self.next_order2key[self.next_order_count] = next_node.name
        self.next_order_count += 1

    # FIXME: check this function
    def add_prune_idx(self, prune_idx, prune_dim):
        if prune_dim == IDX_OUT:
            if self.prune_idx[IDX_OUT] != None:
                self.add_prune_idx_tonext(prune_idx)
            else:
                return False
        elif prune_dim == IDX_IN:
            self.prune_idx[IDX_IN] = prune_idx
            length_prune_idx = len(self.prune_idx[IDX_IN])
            new_prune_idx = self.prune_idx[IDX_IN][: length_prune_idx // self.split]
            self.prune_idx[IDX_OUT] = new_prune_idx
            # self.add_prune_idx_tonext(new_prune_idx)
            self.group.add_prune_idx(new_prune_idx)
        return True


# TODO: finish this
class ChangeNode(BaseNode):  # FIXME: rename
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, None, grad)

    def prune(self):
        pass

    def add_prune_idx(self, prune_idx, prune_dim):
        assert prune_dim == IDX_IN, f"expected dim {IDX_IN}, got {prune_dim}"
        if self.prune_idx[IDX_IN] != None:
            return True
        self.prune_idx[IDX_IN] = prune_idx
        out_prune_idx = prune_idx
        if isinstance(self, (ReshapeNode)):
            out_prune_idx = self.get_out_prune_idx(prune_idx)
        self.prune_idx[IDX_OUT] = prune_idx
        self.add_prune_idx_tonext(out_prune_idx)
        self.pruned = True
        return True

    @abc.abstractmethod
    def get_out_prune_idx(self, in_prune_idx):
        return []


# FIXME: fix the round_to to fix the reshape for rearrange after to_qkv
class ReshapeNode(ChangeNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)
        self.in_shape = list(grad._saved_self_sym_sizes)

    def update_dim_offset(self, dim_offset):
        self.dim_offset_in = dim_offset
        new_dim = self.reshape2dim(self.in_shape, self.out_shape, 1 + dim_offset)
        if self.dim_offset is None:
            self.dim_offset = new_dim - 1
            self.update_next_dim_offset(self.dim_offset)

    def _get_out_shape(self):
        self.out_shape = self.next[0].get_in_shape()  # DONE: maybe bug
        return self.out_shape

    # FIXME: for mvit block
    def get_out_prune_idx(self, in_prune_idx):
        tmp_input = torch.zeros(self.in_shape)
        indexing_tuple = (
            (slice(None),) * (self.dim_offset_in + 1)
            + (in_prune_idx,)
            + (slice(None),) * (len(self.in_shape) - self.dim_offset_in - 2)
        )
        tmp_input[indexing_tuple] = 1
        tmp_input = tmp_input.reshape(self.out_shape)
        out_prune_idx = list(
            set(torch.nonzero(tmp_input)[:, self.dim_offset + 1].tolist())
        )
        self.prune_idx[IDX_OUT] = out_prune_idx
        return self.prune_idx[IDX_IN]

    def reshape2dim(self, in_shape, out_shape, saved_dim):
        # this function aimed to use in_shape and out_shape information
        # to get the dim of the reshape
        # eg. 1. in_shape = [1, 4, 5, 6], out_shape = [1, 4, 30], saved_dim = 1
        #        then the saved_dim not changed, still 1
        # eg. 2. in_shape = [1, 4, 5, 6], out_shape = [1, 20, 6], saved_dim = 3
        #        then the saved_dim changed, from 3 to 2
        # eg. 3. in_shape = [1, 4, 5, 6], out_shape = [20, 6], saved_dim = 3
        #        then the saved_dim changed, from 3 to 1
        saved_shape = in_shape[saved_dim]
        in_shape_len = len(in_shape)
        out_shape_len = len(out_shape)
        assert saved_dim + 1 <= in_shape_len
        prod_saved_dim_left = np.prod(in_shape[:saved_dim])
        prod_saved_dim_right = np.prod(in_shape[saved_dim + 1 :])
        # if saved_dim + 1 <= out_shape_len:
        tmp_out_prod = 1
        i = 0
        while tmp_out_prod < prod_saved_dim_left + 1e-4:
            tmp_out_prod *= out_shape[i]
            i += 1
        # elif saved_dim + 1 > out_shape_len:
        #     tmp_out_prod = 1
        #     i = out_shape_len - 1
        #     while tmp_out_prod < prod_saved_dim_right + 1e-4:
        #         tmp_out_prod *= out_shape[i]
        #         i -= 1
        return i - 1


# TODO: change the dim_offset of next node
class PermuteNode(ChangeNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)
        self.change_dim = tuple(grad._saved_dims)

    def update_dim_offset(self, dim_offset):
        if self.dim_offset is None:
            self.dim_offset = self.change_dim.index(1 + dim_offset) - 1
            self.update_next_dim_offset(self.dim_offset)

    def _get_in_shape(self):
        next_in_shape = self.next[0].get_in_shape()
        self.out_shape = next_in_shape
        self.in_shape = self.__restore_shape(self.change_dim, next_in_shape)
        return self.in_shape

    def _get_out_shape(self):
        prev_out_shape = self.prev[0].get_out_shape()
        self.in_shape = prev_out_shape
        self.out_shape = self.__change_shape(self.change_dim, prev_out_shape)
        return self.out_shape

    def __change_shape(self, change_dim, before_shape):
        after_shape = [0] * len(change_dim)
        for i in range(len(change_dim)):
            after_shape[i] = before_shape[change_dim[i]]
        return after_shape

    def __restore_shape(self, change_dim, after_shape):
        before_shape = [0] * len(change_dim)
        for i in range(len(change_dim)):
            before_shape[change_dim[i]] = after_shape[i]
        return before_shape

    def get_out_prune_idx(self, in_prune_idx):
        return []


class ExpandNode(ChangeNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)
        self.in_shape = list(grad._saved_self_sym_sizes)
        self.out_shape = self.in_shape.copy()

    def get_out_prune_idx(self, in_prune_idx):
        return []


class TransposeNode(ChangeNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)
        self.dim0 = int(grad._saved_dim0)
        self.dim1 = int(grad._saved_dim1)
        self.dim0 = self.dim0 if self.dim0 < 10 else self.dim0 - 18446744073709551616
        self.dim1 = self.dim1 if self.dim1 < 10 else self.dim1 - 18446744073709551616

    def update_dim_offset(self, dim_offset):
        if self.dim_offset is None:
            if self.dim0 == 1 + dim_offset:
                self.dim_offset = self.dim1 - 1
            elif self.dim1 == 1 + dim_offset:
                self.dim_offset = self.dim0 - 1
            else:
                self.dim_offset = 0
            self.update_next_dim_offset(self.dim_offset)

    def _get_in_shape(self):
        next_in_shape = self.next[0].get_in_shape()
        self.out_shape = next_in_shape.copy()
        self.in_shape = next_in_shape.copy()
        self.in_shape[self.dim0] = next_in_shape[self.dim1]
        self.in_shape[self.dim1] = next_in_shape[self.dim0]
        return self.in_shape

    def _get_out_shape(self):
        prev_out_shape = self.prev[0].get_out_shape().copy()
        self.in_shape = prev_out_shape
        self.out_shape = prev_out_shape
        self.out_shape[self.dim0] = prev_out_shape[self.dim1]
        self.out_shape[self.dim1] = prev_out_shape[self.dim0]
        return self.out_shape

    def get_out_prune_idx(self, in_prune_idx):
        return []


class DummyNode(BaseNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, None, grad)
        self.key = True

    def add_prune_idx(self, prune_idx, prune_dim):
        self.add_prune_idx_tonext(prune_idx)
        return True

    def prune(self):
        pass


class OutputNode(DummyNode):
    def __init__(self, name: str, grad, shape) -> None:
        super().__init__(name, grad)
        self.in_shape = list(shape)
        self.out_shape = list(shape)
        self.in_channels = shape[1]
        self.out_channels = shape[1]


class InputNode(DummyNode):
    def __init__(self, name: str, shape) -> None:
        super().__init__(name, None)
        self.in_shape = list(shape)
        self.out_shape = list(shape)
        self.in_channels = shape[1]
        self.out_channels = shape[1]


class PoolNode(BaseNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, None, grad)
        # self.key = True
        self.in_shape = list(grad._saved_self.shape)
        self.in_channels = self.in_shape[1]
        self.out_channels = self.in_channels

    def prune(self):
        pass

    def _get_out_shape(self):
        k1, k2 = self.kernel_size
        s1, s2 = self.stride
        p1, p2 = self.padding
        self.out_shape = self.in_shape.copy()
        self.out_shape[2] = (self.in_shape[2] + 2 * p1 - (k1 - 1) - 1) // s1 + 1
        self.out_shape[3] = (self.in_shape[3] + 2 * p2 - (k2 - 1) - 1) // s2 + 1
        return self.out_shape

    # DONE: check this function
    def add_prune_idx(self, prune_idx, prune_dim):
        assert prune_dim == IDX_IN, f"expected dim {IDX_IN}, got {prune_dim}"
        self.prune_idx[prune_dim] = prune_idx
        self.add_prune_idx_tonext(prune_idx)
        return True


class AdaptiveAvgPoolNode(PoolNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)

    def _get_out_shape(self):
        self.out_shape = self.in_shape.copy()
        self.out_shape[2] = 1
        self.out_shape[3] = 1
        return self.out_shape


class MaxPoolNode(PoolNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)
        self.kernel_size = grad._saved_kernel_size
        self.stride = grad._saved_stride
        self.padding = grad._saved_padding
        self.dilation = grad._saved_dilation


class AvgPoolNode(PoolNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)
        self.kernel_size = grad._saved_kernel_size
        self.stride = grad._saved_stride
        self.padding = grad._saved_padding


class ActivationNode(BaseNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, None, grad)
        self.in_shape = list(grad._saved_self.shape)
        self.in_channels = self.in_shape[1]
        self.out_channels = self.in_channels

    def add_prune_idx(self, prune_idx, prune_dim):
        assert prune_dim == IDX_IN, f"expected dim {IDX_IN}, got {prune_dim}"
        self.prune_idx[prune_dim] = prune_idx
        self.add_prune_idx_tonext(prune_idx)
        return True


class CustomNode(abc.ABC):
    def __init__(self) -> None:
        pass
