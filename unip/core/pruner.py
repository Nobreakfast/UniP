from ..core.node import *
from ..core.group import *
from ..core.algorithm import *
from ..utils.data_type import *

import torch
import torch.nn as nn


def forward_pre_hook(module, input, output):
    if torch.is_tensor(output):
        if not hasattr(output.grad_fn, "metadata"):
            print(module, input[0].shape, output.shape)
            return
        if "module" not in output.grad_fn.metadata:
            output.grad_fn.metadata["module"] = module
        if "output" not in output.grad_fn.metadata:
            output.grad_fn.metadata["output"] = output
        if "input" not in output.grad_fn.metadata:
            output.grad_fn.metadata["input"] = input[0]


def sum_output(output: (torch.Tensor, list, tuple), count: int) -> (torch.Tensor, int):
    total_sum = 0
    if isinstance(output, (list, tuple)):
        for o in output:
            o_total_sum, count = sum_output(o, count)
            total_sum += o_total_sum
    elif isinstance(output, torch.Tensor):
        total_sum += output.sum()
        count += 1
    return total_sum, count


def init_dict_and_list(
    model: nn.Module,
    example_input: (torch.Tensor, list, tuple, dict),
) -> (dict, list, torch.Tensor):
    key2node = {}
    grad_list = []
    input2node = {}
    # get module2key and hooks
    module2key, hooks = get_module2key_and_hooks(model)
    # inference
    if isinstance(example_input, torch.Tensor):
        output = model(example_input)
        node = InputNode("input_0", example_input.shape)
        key2node["input_0"] = node
        input2node[example_input] = node
    elif isinstance(example_input, (list, tuple)):
        output = model(*example_input)
        for i, input in enumerate(example_input):
            node = InputNode(f"input_{i}", input.shape)
            key2node[f"input_{i}"] = node
            input2node[input] = node
    elif isinstance(example_input, dict):
        output = model(**example_input)
        for k, input in example_input.items():
            node = InputNode(f"input_{k}", input.shape)
            key2node[f"input_{k}"] = node
            input2node[input] = node

    # del hooks
    del_hooks(hooks)
    # init output node
    total_sum, count = sum_output(output, 0)
    check_list = [total_sum.grad_fn]
    i = 0
    while i < count:
        grad = check_list[0].next_functions
        check_list.remove(check_list[0])
        for sub_g in grad:
            if sub_g[0].__class__.__name__ == "SumBackward0":
                output_node = OutputNode(
                    f"output_{i}", sub_g[0], sub_g[0]._saved_self_sym_sizes
                )
                key2node[output_node.name] = output_node
                grad_list.append([output_node, sub_g[0].next_functions[0][0]])
                i += 1
            elif sub_g[0].__class__.__name__ == "AddBackward0":
                check_list.append(sub_g[0])
    for i in grad_list:
        print(i[0].name, i[1])

    return module2key, key2node, input2node, grad_list, total_sum


def get_module2key_and_hooks(model):
    module2key = {}
    hooks = []
    for name, module in model.named_modules():
        module2key[module] = name
        if not module._modules:
            hooks.append(module.register_forward_hook(forward_pre_hook))
    return module2key, hooks


def get_param2key(model):
    param2key = {}
    for name, param in model.named_parameters():
        if name.split(".")[-1] in ["weight", "bias"]:
            continue
        param2key[param] = name
    return param2key


def del_hooks(hooks):
    for hook in hooks:
        hook.remove()


def get_group(node, checked_list=[]):
    group = [node]
    # checked_list.append(node)
    if node in checked_list:
        return group, checked_list
    checked_list.append(node)
    search_list = []
    search_list.extend(node.find_next_inin())
    if isinstance(node, InInNode):
        search_list.extend(node.find_prev_inin())
        search_list.extend(node.prev)
    search_list = list(set(search_list))
    for node in search_list:
        if node in checked_list:
            continue
        tmp_g, tmp_c = get_group(node, checked_list)
        group.extend(tmp_g)
        checked_list.extend(tmp_c)
    group = list(set(group))
    checked_list = list(set(checked_list))
    return group, checked_list


def get_groups(key2node):
    groups = []
    search_list = list(key2node.values())
    checked_list = []
    for node in search_list:
        if node.name in checked_list:
            continue
        if isinstance(node, InInNode):
            group_name = []
            tmp_g, _ = get_group(node)
            for node in tmp_g:
                if not node.key:
                    continue
                group_name.append(node.name)
                if isinstance(node, InInNode):
                    group_name.extend([n.name for n in node.find_prev_key()])
            group_name = list(set(group_name))
            # add group
            group = []
            for name in group_name:
                checked_list.append(name)
                group.append(key2node[name])
            groups.append(CurrentGroup(group))

    for node in search_list:
        if not node.key:
            continue
        if node.name in checked_list:
            continue
        groups.append(CurrentGroup([node]))
    return groups


class BasePruner:
    def __init__(
        self,
        model: nn.Module,
        example_input: (torch.Tensor, list, tuple, dict),
        algorithm=RandomAlgo,
    ) -> None:
        self.model = model
        self.example_input = example_input
        self.param2key = get_param2key(model)
        self.backward2node()
        self.update_keynode()
        self.groups = get_groups(self.key2node)
        self.algorithm = algorithm(self.groups, self.key2node)

    def prune(self):
        return self.algorithm.prune()

    def get_g_key(self, last, grad):
        g_name = grad.__class__.__name__
        try:
            g_key = self.module2key[grad.metadata["module"]]
        except:
            g_key = last.name + "." + g_name[:4]
            count = 0
            tmp_g_key = g_key + str(count)
            while tmp_g_key in self.key2node.keys():
                count += 1
                tmp_g_key = g_key + str(count)
            g_key = tmp_g_key
        return g_key

    def grad2node(self, last, grad):
        g_name = grad.__class__.__name__
        if g_name in IGNORE_BACKWARD_TYPE:
            return -1, grad
        elif g_name in PASS_BACKWARD_TYPE:
            return last.name, grad

        if "module" in grad.metadata:
            m = grad.metadata["module"]
            # if isinstance(m, nn.Linear):
            #     print("debug")

        if grad in self.backward2key.keys():
            g_key = self.backward2key[grad]
            node = self.key2node[g_key]
        else:
            g_key = self.get_g_key(last, grad)

            # In-Out
            if g_name == "ConvolutionBackward0":
                if (
                    grad._saved_groups == grad._saved_weight.shape[0]
                    and grad._saved_groups != 1
                ):
                    node = GroupConvNode(g_key, grad.metadata["module"], grad)
                else:
                    node = ConvNode(g_key, grad.metadata["module"], grad)
            elif g_name == "AddmmBackward0":
                node = LinearNode(g_key, grad.metadata["module"], grad)
            # In-In and Out-Out
            elif g_name == "AddBackward0":
                if "module" in grad.metadata:
                    # DONE: Problems, change grad for linear
                    node = LastLinearNode(g_key, grad.metadata["module"], grad)
                    self.backward2key[grad] = g_key
                    self.backward2key[grad.next_functions[0][0]] = g_key
                    self.backward2key[
                        grad.next_functions[0][0].next_functions[0][0]
                    ] = g_key
                    self.backward2key[
                        grad.next_functions[0][0]
                        .next_functions[0][0]
                        .next_functions[0][0]
                    ] = g_key
                    grad = (
                        grad.next_functions[0][0]
                        .next_functions[0][0]
                        .next_functions[0][0]
                    )

                else:
                    node = AddNode(g_key, grad)
                    for sub_g in grad.next_functions:
                        if sub_g[0].__class__.__name__ == "AccumulateGrad":
                            param = sub_g[0].variable
                            param_key = self.param2key[param]
                            node_other = BundleParamNode(param_key, param)
                            self.key2node[node_other.name] = node_other
                            node_other.add_next(node)

            elif g_name == "SubBackward0":
                node = SubNode(g_key, grad)
                for sub_g in grad.next_functions:
                    if sub_g[0].__class__.__name__ == "AccumulateGrad":
                        param = sub_g[0].variable
                        param_key = self.param2key[param]
                        node_other = BundleParamNode(param_key, param)
                        self.key2node[node_other.name] = node_other
                        node_other.add_next(node)
            elif g_name == "MulBackward0":
                node = MulNode(g_key, grad)
                for sub_g in grad.next_functions:
                    if sub_g[0].__class__.__name__ == "AccumulateGrad":
                        # param = grad._saved_other
                        param = sub_g[0].variable
                        param_key = self.param2key[param]
                        node_other = BundleParamNode(param_key, param)
                        self.key2node[node_other.name] = node_other
                        node_other.add_next(node)
            elif g_name == "DivBackward0":
                node = DivNode(g_key, grad)
                param = grad._saved_other
                param_key = self.param2key[param]
                node_other = BundleParamNode(param_key, param)
                self.key2node[node_other.name] = node_other
                node_other.add_next(node)
            elif g_name == "NativeBatchNormBackward0":
                node = BatchNormNode(g_key, grad.metadata["module"], grad)
            elif g_name == "NativeLayerNormBackward0":
                node = LayerNormNode(g_key, grad.metadata["module"], grad)
            elif g_name == "NativeGroupNormBackward0":
                node = GroupNormNode(g_key, grad.metadata["module"], grad)
            # Remap
            elif g_name == "CatBackward0":
                node = ConcatNode(g_key, grad)
            elif g_name == "SplitBackward0":
                node = SplitNode(g_key, grad)
            # Reshape
            elif g_name in RESHAP_BACKWARD_TYPE:
                if "module" in grad.metadata and isinstance(
                    grad.metadata["module"], nn.Linear
                ):
                    # DONE: Problems, change grad for linear
                    node = LastLinearNode(g_key, grad.metadata["module"], grad)
                    self.backward2key[grad] = g_key
                    self.backward2key[grad.next_functions[0][0]] = g_key
                    self.backward2key[
                        grad.next_functions[0][0].next_functions[0][0]
                    ] = g_key
                    grad = grad.next_functions[0][0].next_functions[0][0]
                else:
                    node = ReshapeNode(g_key, grad)
            elif g_name == "PermuteBackward0":
                node = PermuteNode(g_key, grad)
            elif g_name == "ExpandBackward0":
                node = ExpandNode(g_key, grad)
            elif g_name == "TransposeBackward0":
                node = TransposeNode(g_key, grad)
            # elif g_name == "TBackward0":
            #     node = TbackNode(g_key, grad)
            elif g_name == "MeanBackward1":
                node = AdaptiveAvgPoolNode(g_key, grad)
            elif g_name == "MaxPool2DWithIndicesBackward0":
                node = MaxPoolNode(g_key, grad)
            elif g_name == "AvgPool2DBackward0":
                node = AvgPoolNode(g_key, grad)
            elif g_name in ACTIVITION_BACKWARD_TYPE:
                return last.name, grad
            elif g_name in MM_BACKWARD_TYPE:
                node = MatmulNode(g_key, grad)
            elif g_name == "AccumulateGrad":
                if grad.variable in self.input2node.keys():
                    input_node = self.input2node[grad.variable]
                    input_node.add_next(last)
                return last.name, grad
            else:
                print(f"Not supported {g_name}, please add patches")
                return last.name, grad
            self.backward2key[grad] = g_key
            self.key2node[g_key] = node
            if "input" in grad.metadata.keys():
                input = grad.metadata["input"]
                if input in self.input2node.keys():
                    input_node = self.input2node[input]
                    input_node.add_next(node)
        node.add_next(last)
        return g_key, grad

    def backward2node(self):
        """
        Backward to node
        """
        (
            self.module2key,
            self.key2node,
            self.input2node,
            grad_list,
            output_sum,
        ) = init_dict_and_list(self.model, self.example_input)

        checked_list = []
        self.backward2key = {}
        while len(grad_list) > 0:
            [last, grad] = grad_list[0]
            grad_list.remove([last, grad])

            if [last, grad] in checked_list:
                continue
            else:
                checked_list.append([last, grad])

            g_key, grad = self.grad2node(last, grad)
            if g_key == -1:  # IGNORE_BACKWARD_TYPE
                continue

            grad_next = grad.next_functions
            for sub_g in grad_next:
                grad_list.append([self.key2node[g_key], sub_g[0]])

    def update_keynode(self):
        for node in self.key2node.values():
            node.find_next_key()
            node.find_prev_key()
            node.update_shape()
        for node in self.input2node.values():
            dim_map = torch.zeros(tuple(node.in_shape))
            tuple_index = tuple(
                [slice(None) if i == 1 else 0 for i in range(len(node.in_shape))]
            )
            dim_map[tuple_index] = 1
            node.update_next_dim_offset(0, dim_map)
        # print("debug")
