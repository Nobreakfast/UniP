import torch
import torch.nn as nn

import abc


class BaseNode(abc.ABC):
    def __init__(self, name: str, module: (None, nn.Module), grad) -> None:
        self.name = name
        self.module = module
        self.grad = grad
        self.next = []
        self.next_key = []
        self.next_inin = []
        self.prev = []
        self.prev_key = []
        self.prev_inin = []
        self.key = False

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


class InOutNode(BaseNode):
    def __init__(self, name: str, module: nn.Module, grad) -> None:
        super().__init__(name, module, grad)
        self.key = True


class ConvNode(InOutNode):
    def __init__(self, name: str, module: nn.Module, grad) -> None:
        super().__init__(name, module, grad)


class LinearNode(InOutNode):
    # delete the LastLinearNode, as a results, this node need
    # to check the dim of the input and output
    def __init__(self, name: str, module: nn.Module, grad) -> None:
        super().__init__(name, module, grad)


class BundleParamNode(InOutNode):
    def __init__(self, name: str, param) -> None:
        super().__init__(name, None, None)


class OutOutNode(BaseNode):
    def __init__(self, name: str, module: nn.Module, grad) -> None:
        super().__init__(name, module, grad)
        pass


class BatchNormNode(OutOutNode):
    def __init__(self, name: str, module: nn.Module, grad) -> None:
        super().__init__(name, module, grad)


class LayerNormNode(OutOutNode):
    def __init__(self, name: str, module: nn.Module, grad) -> None:
        super().__init__(name, module, grad)


class GroupNormNode(OutOutNode):
    def __init__(self, name: str, module: nn.Module, grad) -> None:
        super().__init__(name, module, grad)


class GroupConvNode(OutOutNode):
    def __init__(self, name: str, module: nn.Module, grad) -> None:
        super().__init__(name, module, grad)


# class LastLinearNode(OutOutNode):
#     def __init__(self, name: str, module: nn.Module, grad) -> None:
#         super().__init__(name, module, grad)


class InInNode(BaseNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, None, grad)
        self.key = True


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


class MatmulNode(InInNode):
    """
    In most cases, this node used in qkv calculation.
    As a results, we do not need to consider the dim
    """

    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)


class RemapNode(BaseNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, None, grad)
        self.key = True


class ConcatNode(RemapNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)
        self.dim = grad._saved_dim


class SplitNode(RemapNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)
        self.in_shape = list(grad._saved_self_sym_sizes)
        self.out_shape = self.in_shape
        self.dim = (
            grad._saved_dim
            if grad._saved_dim < len(self.out_shape)
            else grad._saved_dim - 18446744073709551616
        )
        self.out_shape[self.dim] = grad._saved_split_size


class ChangeNode(BaseNode):  # TODO: rename
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, None, grad)
        self.key = True


class ReshapeNode(ChangeNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)
        self.in_shape = list(grad._saved_self_sym_sizes)


class PermuteNode(ChangeNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)
        self.change_dim = grad._saved_dims
        pass


class ExpandNode(ChangeNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)
        self.in_shape = list(grad._saved_self_sym_sizes)


class TransposeNode(ChangeNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)
        self.dim0 = grad._saved_dim0
        self.dim1 = grad._saved_dim1


class DummyNode(BaseNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, None, grad)
        self.key = True


class OutputNode(DummyNode):
    def __init__(self, name: str, grad, shape) -> None:
        super().__init__(name, grad)
        self.shape = shape


class InputNode(DummyNode):
    def __init__(self, name: str, shape) -> None:
        super().__init__(name, None)
        self.shape = shape


class PoolNode(BaseNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, None, grad)
        self.key = True


class AdaptiveAvgPoolNode(PoolNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)
        pass


class MaxPoolNode(PoolNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)
        pass


class AvgPoolNode(PoolNode):
    def __init__(self, name: str, grad) -> None:
        super().__init__(name, grad)
        pass


class CustomNode(abc.ABC):
    def __init__(self) -> None:
        pass
