from abc import ABC, abstractmethod
import torch as th
import numpy as np
from core.node import Node
from policies.value_transforms import IdentityValueTransform, ValueTransform


def custom_softmax(
    probs: th.Tensor,
    temperature: float | None = None,
    action_mask: th.Tensor | None = None,
) -> th.Tensor:
    
    """
    Applies softmax to the input tensor with a temperature parameter.

    Args:
        probs (th.Tensor): Relative probabilities of actions.
        temperature (float): The temperature parameter. None means dont apply softmax. 0 means stochastic argmax.
        action_mask (th.Tensor, optional): A mask tensor indicating which actions are valid to take. The probability of these should be zero.

    Returns:
        th.Tensor: Probs after applying softmax.
    """

    if temperature is None: # No softmax is applied, returns the full distribution.
        p = probs

    elif temperature == 0.0: # Stochastic argmax
        max_prob = th.max(probs, dim=-1, keepdim=True).values
        p = (probs == max_prob).float()

    else: # Softmax with temperature
        p = th.nn.functional.softmax(probs / temperature, dim=-1)

    if action_mask is not None:
        p[~action_mask] = 0.0

    return p


class Policy(ABC):
    def __call__(self, node: Node) -> int:
        return self.sample(node)

    @abstractmethod
    def sample(self, node: Node) -> int:
        """Take a node and return an action"""

class PolicyDistribution(Policy):

    """
    Also lets us view the full distribution of the policy, not only sample from it.
    When we have the distribution, we can choose how to sample from it.
    We can either sample stochasticly from distribution or deterministically choose the action with the highest probability.
    We can also apply softmax with temperature to the distribution.
    """

    temperature: float
    value_transform: ValueTransform

    def __init__(
        self,
        temperature: float = None,
        value_transform: ValueTransform = IdentityValueTransform,
    ) -> None:
        
        super().__init__()
        
        self.temperature = temperature
        self.value_transform = value_transform

    def sample(self, node: Node) -> int:
        """
        Returns a random action from the distribution
        """
        return int(self.softmaxed_distribution(node).sample().item())

    @abstractmethod
    def _probs(self, node: Node) -> th.Tensor:
        """
        Returns the relative probabilities of the actions (excluding the special action)
        """
        pass

    def self_prob(self, node: Node, probs: th.Tensor) -> float:

        """
        Returns the relative probability of selecting the node itself
        """

        return probs.sum() / (node.visits - 1)

    def add_self_to_probs(self, node: Node, probs: th.Tensor) -> th.Tensor:
        
        """
        Takes the current policy and adds one extra value to it, which is the probability of selecting the node itself.
        Should return a tensor with one extra value at the end
        The default choice is to set it to 1/visits
        Note that policy is not yet normalized, so we can't just add 1/visits to the last value
        """

        self_prob = self.self_prob(node, probs)
        return th.cat([probs, th.tensor([self_prob])])

    def softmaxed_distribution(
        self, node: Node, include_self=False
    ) -> th.distributions.Categorical:
        
        """
        Returns the distribution over actions according to the policy,
        applying the temperature parameter in custom_softmax().
        """

        # Policy for leaf nodes: always select the special action (a_v)
        # Policy tensor will be e.g. [0, 0, 0, 0, 1]
        if include_self and len(node.children) == 0:
            probs = th.zeros(int(node.action_space.n) + include_self, dtype=th.float32)
            probs[-1] = 1.0
            return th.distributions.Categorical(probs=probs)

        # Get the "raw" probability logits
        probs = self._probs(node) 

        # Softmax the logits. This applies the temperature parameter.
        # Note that this is not applied to the probability of a_v.
        softmaxed_probs = custom_softmax(probs, self.temperature, None)

        # Add the self probability. 
        if include_self:
            softmaxed_probs = self.add_self_to_probs(node, softmaxed_probs)

        return th.distributions.Categorical(probs=softmaxed_probs)


class RandomPolicy(Policy):
    def sample(self, node: Node) -> int:
        return node.action_space.sample()
