import graphviz
import sys
sys.path.append('src/')
from core.node import Node
import os

def visualize_tree(tree: Node, filename: str, var_fn=None, max_depth=None):
    """
    Visualize a single tree using graphviz.

    Args:
        tree (Node): The root node of the tree to visualize.
        filename (str): The filename to save the visualization.
        var_fn (Optional[Callable[[Node], Any]]): A function to extract additional information from nodes.
        max_depth (Optional[int]): The maximum depth to visualize.
    """
    dot = graphviz.Digraph(comment="MCTS Tree")
    tree._add_node_to_graph(dot, var_fn, max_depth=max_depth)
    dot.render(filename=filename, view=False)

def visualize_trees(trees, output_dir, var_fn=None, max_depth=None, max_trees=100):
    """
    Visualize multiple trees and save them to the specified directory.

    Args:
        trees (List[Node]): A list of root nodes of the trees to visualize.
        output_dir (str): The directory to save the visualizations.
        var_fn (Optional[Callable[[Node], Any]]): A function to extract additional information from nodes.
        max_depth (Optional[int]): The maximum depth to visualize.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, tree in enumerate(trees):
        if max_trees is not None and i >= max_trees:
            break
        filename = os.path.join(output_dir, f"tree_envstep_{i}.gv")
        visualize_tree(tree, filename, var_fn, max_depth)

