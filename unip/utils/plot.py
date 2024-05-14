from graphviz import Digraph
import matplotlib.pyplot as plt
import os
import time

from unip.core.node import *


def plot_graph(graph: dict, display=True, save_path=f"logs/plot/fig{time.time()}"):
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
    if display:
        dot.view()
    return dot
