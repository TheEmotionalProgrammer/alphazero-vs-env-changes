from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import torch as th
import numpy as np
from node import Node

ObservationType = TypeVar("ObservationType")



class Policy(ABC, Generic[ObservationType]):

    def __call__(self, node: Node[ObservationType]) -> np.int64:
        return self.sample(node)


    @abstractmethod
    def sample(self, node: Node[ObservationType]) -> np.int64:
        """Take a node and return an action"""

class PolicyDistribution(Policy[ObservationType]):
    """Also lets us view the full distribution of the policy, not only sample from it.
    When we have the distribution, we can choose how to sample from it.
    We can either sample stochasticly from distribution or deterministically choose the action with the highest probability.
    We can also apply softmax with temperature to the distribution.
    """

    def sample(self, node: Node[ObservationType]) -> th.Tensor:
        """
        Returns a random action from the distribution
        """
        return self.distribution(node).sample()

    @abstractmethod
    def distribution(self, node: Node[ObservationType]) -> th.distributions.Categorical:
        """The distribution of the policy. Must sum to 1 and be all positive."""
        pass



class OptionalPolicy(ABC, Generic[ObservationType]):
    def __call__(self, node: Node[ObservationType]) -> np.int64 | None:
        return self.sample(node)


    @abstractmethod
    def sample(self, node: Node[ObservationType]) -> np.int64 | None:
        """Take a node and return an action"""





class UCT(OptionalPolicy[ObservationType]):
    def __init__(self, c: float):
        self.c = c

    def sample(self, node: Node[ObservationType]) -> np.int64 | None:
        # if not fully expanded, return None
        if not node.is_fully_expanded():
            return None

        # if fully expanded, return the action with the highest UCB value
        # Idea: potentially mess around with making this stochastic
        return max(node.children, key=lambda action: self.ucb(node, action))

    def ucb(self, node: Node[ObservationType], action: np.int64) -> float:
        child = node.children[action]
        return child.default_value() + self.c * (node.visits / child.visits) ** 0.5



class PUCT(OptionalPolicy[ObservationType]):
    def __init__(self, c: float):
        self.c = c

    def sample(self, node: Node) -> np.int64 | None:
        # if not fully expanded, return None
        if not node.is_fully_expanded():
            return None

        # if fully expanded, return the action with the highest UCB value
        # Idea: potentially mess around with making this stochastic
        return max(node.children, key=lambda action: self.puct(node, action))

    # TODO: this can def be sped up (calculate the denominator once)
    def puct(self, node: Node, action: np.int64) -> float:
        child = node.children[action]
        prior = node.prior_policy[action]
        return child.default_value() + self.c * prior * (node.visits ** 0.5) / (child.visits + 1)



class RandomPolicy(Policy[ObservationType]):
    def sample(self, node: Node[ObservationType]) -> np.int64:
        return node.action_space.sample()


class DefaultExpansionPolicy(Policy[ObservationType]):
    def sample(self, node: Node[ObservationType]) -> np.int64:
        # returns a uniformly random unexpanded action
        return node.sample_unexplored_action()


class DefaultTreeEvaluator(PolicyDistribution[ObservationType]):
    # the default tree evaluator selects the action with the most visits
    def distribution(self, node: Node[ObservationType]) -> th.distributions.Categorical:
        visits = th.zeros(int(node.action_space.n), dtype=th.float32)
        for action, child in node.children.items():
            visits[action] = child.visits

        return th.distributions.Categorical(visits)



class SoftmaxDefaultTreeEvaluator(PolicyDistribution[ObservationType]):
    """
    Same as DefaultTreeEvaluator but with softmax applied to the visits. temperature controls the softmax temperature.
    """

    def __init__(self, temperature: float):
        self.temperature = temperature
    # the default tree evaluator selects the action with the most visits
    def distribution(self, node: Node[ObservationType]) -> th.distributions.Categorical:
        visits = th.zeros(int(node.action_space.n), dtype=th.float32)
        for action, child in node.children.items():
            visits[action] = child.visits

        return th.distributions.Categorical(th.softmax(visits / self.temperature, dim=-1))

