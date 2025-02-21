from collections import Counter
from typing import Dict, Generic, List, TypeVar, Optional, Any, Callable, Tuple
import gymnasium as gym
import numpy as np
import torch as th
import graphviz
from environments.frozenlake.frozen_lake import actions_dict

ObservationType = TypeVar("ObservationType")

NodeType = TypeVar("NodeType", bound="Node")

class Node(Generic[ObservationType]):

    parent: Optional["Node[ObservationType]"]
    children: Dict[int, "Node[ObservationType]"]
    visits: int = 0
    subtree_sum: float = 0.0  # sum of reward and value of all children
    value_evaluation: float  # Expected future reward
    reward: float  # Reward received when stepping into this node
    action_space: gym.spaces.Discrete  
    observation: Optional[ObservationType]
    prior_policy: th.Tensor
    env: Optional[gym.Env]
    variance: float | None = None
    policy_value: float | None = None
    
    def __init__(
        self,
        env: gym.Env,
        parent: Optional["Node[ObservationType]"],
        reward: float,
        action_space: gym.spaces.Discrete,
        observation: Optional[ObservationType],
        terminal: bool = False,
        ncols: int = 8,
    ):
        
        self.children = {} # dictionary of children where key is action, value is the child node
        self.action_space = action_space
        self.reward = reward
        self.parent = parent
        self.terminal = terminal
        self.observation = observation
        self.env = env
        self.problematic = False

        self.ncols = ncols

    def coords(self, observ):
        return (observ // self.ncols, observ % self.ncols) if observ is not None else None

    def is_terminal(self) -> bool:
        return self.terminal

    def step(self, action: int) -> "Node[ObservationType]":

        """
        Return the child node after taking the given action. 
        The child has to be expanded already, otherwise the method will throw an error.
        """
        
        child = self.children[action]
        return child

    def default_value(self) -> float:

        """
        The default value estimate for taking this action is the average of the rewards + value estimates of all children
        """

        return self.subtree_sum / self.visits

    def is_fully_expanded(self) -> bool:

        """
        Returns True if all possible actions have been expanded.
        """

        return len(self.children) == self.action_space.n

    def sample_unexplored_action(self) -> int:

        """
        An optional mask which indicates if an action can be selected. 
        Expected np.ndarray of shape (n,) and dtype np.int8 where 1 represents valid actions and 0 invalid / infeasible actions. 
        If there are no possible actions (i.e. np.all(mask == 0)) then space.start will be returned.
        """

        mask = np.ones(self.action_space.n, dtype=np.int8)
        for action in self.children:
            mask[action] = 0
        return self.action_space.sample(mask=mask)

    def get_root(self) -> "Node[ObservationType]":

        """
        Returns the root node of the tree associated with this node.
        """

        node: Node[ObservationType] | None = self

        while node.parent is not None:
            node = node.parent
        return node


    def visualize(
        self,
        var_fn: Optional[Callable[["Node[ObservationType]"], Any]] = None,
        max_depth: Optional[int] = None,
    ) -> None:
        
        dot = graphviz.Digraph(comment="Planning Tree")
        self._add_node_to_graph(dot, var_fn, max_depth=max_depth)
        dot.render(filename="plan_tree.gv", view=True)


    def _add_node_to_graph(
        self,
        dot,
        var_fn: Optional[Callable[["Node[ObservationType]"], Any]] = None,
        max_depth: Optional[int] = None,
    ) -> None:
        if max_depth is not None and max_depth == 0:
            return
        label = f"O: {self.coords(self.observation)}, R: {self.reward}, MS: {self.default_value(): .2f}, V: {self.value_evaluation: .2f}\nVisit: {self.visits}, T: {int(self.terminal)}"
        if var_fn is not None:
            label += f", VarFn: {var_fn(self)}"

        if self.policy_value is not None:
            label += f", PV: {self.policy_value: .2f}"
        if self.variance is not None:
            label += f", Var: {self.variance: .2f}"
        dot.node(str(id(self)), label=label)
        for action, child in self.children.items():
            child._add_node_to_graph(
                dot, var_fn, max_depth=max_depth - 1 if max_depth is not None else None
            )

            dot.edge(str(id(self)), str(id(child)), label=f"Action: {actions_dict[action]}")


    def state_visitation_counts(self) -> Counter:

        """
        Returns a counter of the number of times each state has been visited
        """

        counter = Counter()
        # add the current node
        counter[self.observation] = self.visits if self.is_terminal() else 1
        # add all children
        for child in self.children.values():
            counter.update(child.state_visitation_counts())

        return counter

    def get_children(self):

        """
        Returns the list of children of the node.
        """

        l: List[Node | None] = [None] * self.action_space.n
        for key, child in self.children.items():
            l[key] = child
        return l

    def reset_policy_value(self):
        
        """
        Reset policy estimates of the node and the whole subtree.
        """

        self.policy_value = None
        for child in self.children.values():
            child.reset_policy_value()

    def reset_variance(self):
        
        """
        Reset variance estimates of the node and the whole subtree.
        """
    
        self.variance = None
        for child in self.children.values():
            child.reset_variance()

    def reset_var_val(self):

        """
        Reset value and variance estimates of the node and the whole subtree.
        """

        self.variance = None
        self.policy_value = None
        for child in self.children.values():
            child.reset_var_val()

    def reset_visits(self):
        self.visits = 1
        for child in self.children.values():
            child.visits = 1
            child.reset_visits()

    def __str__(self):
        return f"Visits: {self.visits}, ter: {int(self.terminal)}\nR: {self.reward}\n Value_Estimate: {self.value_evaluation}, Mean_Value: {self.default_value()}"

    def __repr__(self):
        return self.__str__()
