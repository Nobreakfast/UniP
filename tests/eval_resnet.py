import torch
from torchvision.models import resnet18

from unip.core.graph import BackwardGrapher
from unip.core.group import AddGrouper
from unip.core.algorithm import UniformAlgorithm
from unip.core.node import *
from unip.utils.evaluation import cal_flops
from unip.utils.plot import plot_graph

def print_graph(graph):
    for name, node in graph.items():
        if isinstance(node, ModuleNode):
            print(name, node.module)
def eva_resnet18():
    model = resnet18()
    example_input = torch.rand(1, 3, 224, 224)
    cal_flops(model, example_input, device="cpu")
    graph = BackwardGrapher(model, example_input).graph
    # plot_graph(graph)
    group = AddGrouper(model, example_input, graph).group
    # for g in group:
    #     print([n.name for n in g.nodes])
    # print_graph(graph)
    algorithm = UniformAlgorithm(group, 0.7)
    # algorithm._prune()
    # print_graph(graph)
    cal_flops(model, example_input, device="cpu")
    output = model(example_input)
    print(output.shape)

if __name__ == "__main__":
    eva_resnet18()