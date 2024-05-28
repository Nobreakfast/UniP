from graphviz import Digraph

from unip.core.node import *


def plot_graph(graph: dict, display: bool = True, save_path: str = None):
    dot = Digraph(comment="Graph of Model")
    for name, node in graph.items():
        if isinstance(node, ModuleNode):
            shape = "box"
            color = "lightblue"
        elif isinstance(node, ActionNode):
            shape = "diamond"
            color = "lightcoral"
        elif isinstance(node, ParamNode):
            shape = "ellipse"
            color = "lightgreen"
        dot.node(
            node.name,
            node.__class__.__name__ + ": " + node.name,
            shape=shape,
            color=color,
        )

    for name, node in graph.items():
        for next_node in node.next:
            dot.edge(node.name, next_node.name)
        # for prev_node in node.prev:
        #     dot.edge(node.name, prev_node.name)

    dot.render(save_path, format="pdf")
    print(f"Graph has been saved to {save_path}")
    if display:
        dot.view()
    return dot


def plot_group(groups: dict, display: bool = True, save_path: str = None):
    pass
