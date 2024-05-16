import torch

from unip.core.graph import BackwardGrapher
from unip.core.group import AddGrouper
from unip.core.algorithm import UniformAlgorithm
from unip.core.node import *
from unip.utils.plot import plot_graph
from unip.utils.evaluation import cal_flops

def print_graph(graph):
    for name, node in graph.items():
        if isinstance(node, ModuleNode):
            print(name, node.module)
def eval_split():
    class SplitModule(nn.Module):
        def __init__(self):
            super(SplitModule, self).__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
            self.conv2_1 = nn.Conv2d(8, 8, 3, 1, 1)
            self.conv2_2 = nn.Conv2d(8, 8, 3, 1, 1)
            self.conv3_1 = nn.Conv2d(8, 8, 3, 1, 1)
            self.conv3_2 = nn.Conv2d(8, 8, 3, 1, 1)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(8*32*32, 10)

        def forward(self, x):
            x = self.conv1(x)
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1 = self.conv2_1(x1)
            x2 = self.conv2_2(x2)
            x1 = self.conv3_1(x1)
            x2 = self.conv3_2(x2)
            x = torch.cat([x1, x2], dim=1)
            x = self.flatten(x)
            x = self.fc(x)
            return x
    model = SplitModule()
    example_input = torch.rand(1, 3, 32, 32)
    cal_flops(model, example_input, device="cpu")
    graph = BackwardGrapher(model, example_input).graph
    group = AddGrouper(model, example_input, graph).group
    algorithm = UniformAlgorithm(group, 0.7)
    cal_flops(model, example_input, device="cpu")
    output = model(example_input)
    print(output.shape)

if __name__ == "__main__":
    eval_split()