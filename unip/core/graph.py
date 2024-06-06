import time

import torch
import torch.nn as nn
import abc
from einops import rearrange
from graphviz import Digraph
import logging

from unip.core.node import *
from unip.utils.data_type import *
from unip.utils.plot import plot_graph
from unip.utils.evaluation import to_device, process_data

logger = logging.getLogger("[Graph:")


def name2grapher(name):
    if name == "backward":
        return BackwardGrapher
    else:
        raise ValueError(
            f"Unsupported grapher name: {name}. \
            Please use 'backward'. \
            Or leave issue at https://github.com/Nobreakfast/UniP/issues/new/choose"
        )


def find_module_input_grad(gradfn, aim_gradfn):
    checkin_list = [[sub_g[0], gradfn] for sub_g in gradfn.next_functions]
    while checkin_list:
        [sub_g, gradfn] = checkin_list.pop()
        if sub_g == aim_gradfn:
            return gradfn
        if not hasattr(sub_g, "next_functions"):
            continue
        checkin_list.extend(
            [[sub_sub_g[0], sub_g] for sub_sub_g in sub_g.next_functions]
        )
    return None


def _forward_hook(module, input, output):
    if not torch.is_tensor(output):
        output = output[0]
    if not hasattr(output.grad_fn, "metadata"):
        print(module, input[0].shape, output.shape)
        return
    if "module" not in output.grad_fn.metadata:
        output.grad_fn.metadata["module"] = module
    if "output" not in output.grad_fn.metadata:
        output.grad_fn.metadata["output"] = output
    if "input" not in output.grad_fn.metadata:
        output.grad_fn.metadata["input"] = input[0]


def _del_hook(hooks):
    for hook in hooks:
        hook.remove()


def _get_module2key(module, ignore_modules=None):
    module2key = {}
    hooks = []
    for name, module in module.named_modules():
        module2key[module] = name
        if ignore_modules is not None and name in ignore_modules.keys():
            hooks.append(module.register_forward_hook(_forward_hook))
        if not module._modules:
            hooks.append(module.register_forward_hook(_forward_hook))
    return module2key, hooks


def _process_input(data, name=input, count=0):
    """
    input data: torch.Tensor, tuple, list, dict
    output data: original data, name to data dict
    """
    if isinstance(data, torch.Tensor):
        # input_dict = {f"input_{count}": data}
        data = to_device(data)
        return data, {f"input_{count}": data}
    elif isinstance(data, (tuple, list)):
        input_dict = {}
        for i, d in enumerate(data):
            d, sub_input_dict = _process_input(d, f"{name}_{count}", i)
            if d is None:
                continue
            input_dict.update(sub_input_dict)
        return data, input_dict
    elif isinstance(data, dict):
        input_dict = {}
        for i, v in enumerate(data.values()):
            v, sub_input_dict = _process_input(v, f"{name}_{count}", i)
            if v is None:
                continue
            input_dict.update(sub_input_dict)
        return data, input_dict
    return data, {}


def _process_output(data, name="output"):
    output_dict = {}
    sum = 0
    if isinstance(data, torch.Tensor):
        output_dict[f"{name}"] = data
        sum += data.sum()
    elif isinstance(data, (tuple, list)):
        for i, sub_out in enumerate(data):
            sub_out, sub_sum = _process_output(sub_out, f"{name}_{i}")
            output_dict.update(sub_out)
            sum += sub_sum
    elif isinstance(data, dict):
        for i, v in enumerate(data.values()):
            v, sub_sum = _process_output(v, f"{name}_{i}")
            output_dict.update(v)
            sum += sub_sum
    return output_dict, sum


def _get_input_node(data):
    input_dict = {}
    param2name = {}
    for name, data in data.items():
        input_dict[name] = InputNode(name, data)
        param2name[data] = name
    return input_dict, param2name


def _get_output_node(data):
    output_dict = {}
    for name, data in data.items():
        output_dict[name] = OutputNode(name, data.grad_fn, data)
    return output_dict


def _get_param_node(model):
    param_dict = {}
    param2name = {}
    for name, param in model.named_parameters():
        if name[-4:] == "bias":
            continue
        if name[-6:] == "weight":
            continue
        param_dict[name] = BundleNode(name, param)
        param2name[param] = name
    return param_dict, param2name


class BaseGrapher(abc.ABC):
    def __init__(self):
        self._graph = None

    @property
    def graph(self):
        if self._graph is None:
            logger.info(f"{self.__class__.__name__}] Building Graph...")
            self._graph = self._build_graph()
            logger.info(f"{self.__class__.__name__}] Graph Built.")
        return self._graph

    @abc.abstractmethod
    def _build_graph(self):
        pass

    def plot(self, display=True, save_path=None):
        if save_path is None:
            save_path = f"logs/plot/fig_{time.time()}"
            logger.info(f"{self.__class__.__name__}] Save plot to {save_path}")
        plot_graph(self.graph, display=display, save_path=save_path)


class BackwardGrapher(BaseGrapher):
    def __init__(
        self,
        model: nn.Module,
        example_input: (torch.Tensor, tuple, list, dict),
        ignore_modules=None,
    ):
        super().__init__()
        self.ignore_modules = ignore_modules
        self.model = model
        example_input = process_data(example_input)
        self.example_input, self.input_dict = _process_input(example_input)
        self.backward2name = {}
        self.module2name = {}
        self.name2node = {}
        self.param2name = {}

    def _build_graph(self):
        module2name, hooks = _get_module2key(self.model, self.ignore_modules)
        self.module2name = module2name
        if isinstance(self.example_input, dict):
            self.output, sum = _process_output(self.model(**self.example_input))
        elif isinstance(self.example_input, (tuple, list)):
            self.output, sum = _process_output(self.model(*self.example_input))
        else:
            self.output, sum = _process_output(self.model(self.example_input))
        _del_hook(hooks)
        sum.backward(retain_graph=True)
        self._update_inout_dict()
        self.get_gradfn_list(self.onode_dict)
        # TODO: whether the input has been connected. if not, the multi-modality pruner may failed
        self.check_input()
        return self.name2node

    def _update_inout_dict(self):
        self.inode_dict, param2name = _get_input_node(self.input_dict)
        self.param2name.update(param2name)
        self.onode_dict = _get_output_node(self.output)
        self.param_dict, param2name = _get_param_node(self.model)
        self.param2name.update(param2name)
        self.name2node.update(self.inode_dict)
        self.name2node.update(self.onode_dict)
        self.name2node.update(self.param_dict)

    def get_gradfn_list(self, onode_dict):
        # init the list
        checkin_list = []
        checkout_list = []
        # get the output gradfn
        for node in onode_dict.values():
            if node.gradfn is None:
                continue
            checkin_list.append([node, node.gradfn])

        # checkout the backward graph
        while checkin_list:
            node = None
            last_node, gradfn = checkin_list.pop()
            gradfn_name = gradfn.__class__.__name__
            logger.info(
                f"{self.__class__.__name__}] Pop ==> [{last_node.name}, {gradfn_name}] from checkin_list"
            )
            # check if the combination of module and gradfn has been record
            if [last_node, gradfn] in checkout_list:
                continue

            gradfn_next = gradfn.next_functions
            if gradfn in self.backward2name.keys():
                # if the node has been created, get node from name
                node = self.name2node[self.backward2name[gradfn]]
            else:
                if "module" in gradfn.metadata:
                    # if it is a module, get node from module
                    module = gradfn.metadata["module"]

                    # if module in self.ignore_modules.keys():
                    #     if self.ignore_modules[module] is not None:
                    #         node = self.ignore_modules[module](
                    #             self.module2name[module], module, gradfn
                    #         )
                    #     else:
                    #         node = IgnoreNode(self.module2name[module], module, gradfn)
                    #     gradfn_next_prev = find_module_input_grad(
                    #         gradfn, gradfn.metadata["input"].grad_fn
                    #     )
                    #     gradfn_next = gradfn_next_prev.next_functions
                    if module in self.module2name:
                        name = self.module2name[module]
                        if name in self.ignore_modules.keys():
                            if self.ignore_modules[name] is not None:
                                node = self.ignore_modules[name](name, module, gradfn)
                            else:
                                node = IgnoreNode(name, module, gradfn)
                            gradfn_next_prev = find_module_input_grad(
                                gradfn, gradfn.metadata["input"].grad_fn
                            )
                            gradfn_next = gradfn_next_prev.next_functions
                        else:
                            # InOut
                            if isinstance(module, CONV_TYPE):
                                node = ConvNode(name, module, gradfn)
                            elif isinstance(module, LINEAR_TYPE):
                                if len(gradfn.metadata["input"].shape) > 2:
                                    node = LastLinearNode(name, module, gradfn)
                                    try:
                                        gradfn_next = (
                                            gradfn.next_functions[0][0]
                                            .next_functions[0][0]
                                            .next_functions[0][0]
                                            .next_functions
                                        )
                                    except:
                                        gradfn_next = (
                                            gradfn.next_functions[0][0]
                                            .next_functions[1][0]
                                            .next_functions
                                        )
                                else:
                                    node = LinearNode(name, module, gradfn)
                            elif isinstance(module, nn.Embedding):
                                node = EmbeddingNode(name, module, gradfn)
                            # Norm
                            elif isinstance(module, NORM_TYPE):
                                node = NormNode(name, module, gradfn)
                            # Activation
                            elif isinstance(module, ACTIVITION_TYPE):
                                count = 0
                                name = self.module2name[module] + "_" + str(count)
                                while name in self.name2node.keys():
                                    count += 1
                                    name = self.module2name[module] + "_" + str(count)
                                node = ActivationNode(name, module, gradfn)
                            # Pooling
                            elif isinstance(module, POOLING_TYPE):
                                node = PoolNode(name, module, gradfn)
                            # DimSwitch
                            elif isinstance(module, nn.Flatten):
                                node = FlattenNode(name, module, gradfn)
                            else:
                                logger.warning(
                                    f"Unknown module: {module}, skip! \
                                    Please leave issue at https://github.com/Nobreakfast/UniP/issues/new/choose"
                                )
                if node is None:
                    # if it is not a module, get node from gradfn type
                    node_name = gradfn_name[:3] + "_" + last_node.name
                    if gradfn_name in ACTIVITION_BACKWARD_TYPE:
                        # check if it is a activation function
                        node = ActivationNode(node_name, None, gradfn)
                    # InIn
                    elif gradfn_name in ADD_BACKWARD_TYPE:
                        # check if it is a add function
                        node = AddNode(node_name, gradfn)
                    elif gradfn_name in MM_BACKWARD_TYPE:
                        # check if it is a mul function
                        node = MatMulNode(node_name, gradfn)

                    # Reshape
                    elif gradfn_name in RESHAP_BACKWARD_TYPE:
                        # check if it is a reshape function
                        # chcek if it is a rearrange function
                        if (
                            gradfn_next[0][0].__class__.__name__ == "PermuteBackward0"
                            and gradfn_next[0][0]
                            .next_functions[0][0]
                            .__class__.__name__
                            in RESHAP_BACKWARD_TYPE
                        ):
                            node = RearrangeNode(node_name, gradfn)
                            gradfn_next = (
                                gradfn_next[0][0].next_functions[0][0].next_functions
                            )
                        else:
                            node = ReshapeNode(node_name, None, gradfn)

                    # Remap
                    elif gradfn_name == "CatBackward0":
                        # check if it is a cat function
                        node = ConcatNode(node_name, gradfn)
                    elif gradfn_name == "SplitBackward0":
                        # check if it is a split function
                        node = SplitNode(node_name, gradfn)
                    elif gradfn_name == "RepeatBackward0":
                        # check if it is a repeat function
                        node = RepeatNode(node_name, gradfn)
                    elif gradfn_name == "IndexBackward0":
                        # check if it is a index select function
                        node = IndexSelectNode(node_name, gradfn)
                    elif gradfn_name == "SliceBackward0":
                        # check if it is a slice function
                        node = SliceNode(node_name, gradfn)
                    elif gradfn_name == "CopySlices":
                        # TODO: check if it is a copy slices function
                        node = CopySlicesNode(node_name, gradfn)
                    elif gradfn_name == "StackBackward0":
                        # TODO: check if it is a stack function
                        node = StackNode(node_name, gradfn)
                    elif gradfn_name == "MaxBackward0":
                        # check if it is a max function
                        node = MaxNode(node_name, gradfn)

                    # MapChange
                    elif gradfn_name in POOLING_BACKWARD_TYPE:
                        # check if it is a pooling function
                        node = PoolNode(node_name, None, gradfn)
                    elif gradfn_name == "CloneBackward0":
                        node = CloneNode(node_name, gradfn)

                    # DimSwitch
                    elif gradfn_name == "PermuteBackward0":
                        # check if it is a permute function
                        node = PermuteNode(node_name, gradfn)
                    elif gradfn_name == "TransposeBackward0":
                        # check if it is a transpose function
                        node = TransposeNode(node_name, gradfn)
                    elif gradfn_name == "TBackward0":
                        # TODO: check if it is a t function
                        node = TransposeNode(node_name, gradfn)
                    elif gradfn_name == "ReshapeAliasBackward0":
                        # check if it is a flatten function
                        node = FlattenNode(node_name, None, gradfn)
                    elif gradfn_name == "SqueezeBackward1":
                        node = SqueezeNode(node_name, gradfn)
                    else:
                        logger.warning(
                            f"{self.__class__.__name__}] Unknown gradfn: {gradfn_name}, skip! \
                            Please leave issue at https://github.com/Nobreakfast/UniP/issues/new/choose"
                        )

                if node is not None:
                    self.backward2name[gradfn] = node.name
                    self.name2node[node.name] = node
                else:
                    node = last_node

                # search next gradfn
                for sub_gradfn in gradfn_next:
                    if sub_gradfn[0] is None:
                        continue
                    sub_gradfn_name = sub_gradfn[0].__class__.__name__
                    if sub_gradfn_name == "AccumulateGrad":
                        if sub_gradfn[0].variable in self.param2name.keys():
                            sub_node = self.name2node[
                                self.param2name[sub_gradfn[0].variable]
                            ]
                            sub_node.add_next(node)
                            node.add_next(last_node)
                        continue
                    elif sub_gradfn_name in IGNORE_BACKWARD_TYPE:
                        continue
                    checkin_list.append([node, sub_gradfn[0]])
                    # print(f"Add +++ [{node.name}, {sub_gradfn_name}] to checkin_list")
                    logger.info(
                        f"{self.__class__.__name__}] Add +++ [{node.name}, {sub_gradfn_name}] to checkin_list"
                    )

                # connect current node to last node
                if last_node.name == node.name:
                    continue

            node.add_next(last_node)

    def print_graph(self, forward=True):
        node_dict = self.inode_dict if forward else self.onode_dict
        for node in node_dict.values():
            node.print(forward=forward)

    def check_input(self):
        for node in self.inode_dict.values():
            pass


if __name__ == "__main__":

    class TestModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.non_param = 2
            self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(8)
            self.p1 = nn.Parameter(torch.randn(1, 1, 1, 4))
            self.conv2 = nn.Conv2d(4, 8, 3, 1, 1)
            self.conv3 = nn.Conv2d(4, 8, 3, 1, 1)
            self.bn23 = nn.BatchNorm2d(16)
            self.conv4 = nn.Conv2d(16, 16, 3, 1, 1, groups=16)
            self.bn4 = nn.BatchNorm2d(16)
            self.conv5_1 = nn.Conv2d(16, 16, 3, 1, 1)
            self.conv5_2 = nn.Conv2d(16, 16, 3, 1, 1)
            self.identity = nn.Identity()
            self.bn5i = nn.BatchNorm2d(16)
            self.bn45 = nn.BatchNorm2d(32)
            self.fcp = nn.Linear(32, 32)
            self.conv6 = nn.Conv2d(32, 1, 3, 1, 1)
            self.bn6 = nn.BatchNorm2d(1)
            self.p6 = nn.Parameter(torch.randn(1, 1, 1, 1))
            # self.pool = nn.MaxPool2d(2, 2)
            self.pool = nn.AvgPool2d(2, 2)
            self.flat = nn.Flatten()
            self.fc = nn.Linear(4, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = x + self.p1
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1 = self.conv2(x1)
            x2 = self.conv3(x2)
            x = torch.cat([x1, x2], dim=1)
            x = self.bn23(x)
            identity = self.identity(x)
            x1 = self.conv4(x)
            x1 = self.bn4(x1)
            x2 = self.conv5_1(x)
            x2 = self.conv5_2(x2 + identity)
            x2 = self.bn5i(x2)
            x = torch.cat([x1, x2], dim=1)
            x = self.bn45(x)  # (1, 32, 4, 4)
            x = rearrange(x, "b c h w -> b (h w) c")
            x = self.fcp(x)  # (1, 4, 4, 32) -> (1, 4, 4, 32)
            x = rearrange(x, "b (h w) c -> b c h w", h=4)
            x = self.conv6(x)  # (1, 32, 4, 4)
            x = self.bn6(x)
            x = torch.nn.functional.relu(x)
            x = x * self.p6
            x = self.pool(x)
            x = self.flat(x)
            x = self.fc(x) * self.non_param
            return x

    model = TestModel()
    example_input = torch.randn(1, 3, 4, 4)
    grapher = BackwardGrapher(model, example_input)
    graph = grapher.graph
    plot_graph(graph)
