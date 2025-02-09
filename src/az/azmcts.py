import torch as th

from core.mcts import MCTS
from az.model import AlphaZeroModel
from core.node import Node
from policies.policies import Policy
from environments.frozenlake.utils import fz_compute_distances, fz_perfect_value


class AlphaZeroMCTS(MCTS):

    """
    Implementation of the AlphaZero Monte Carlo Tree Search (AZMCTS) algorithm.

    Attributes:
        model (AlphaZeroModel): The AlphaZero model used for value and policy prediction.
        dir_epsilon (float): The epsilon value for adding Dirichlet noise to the prior policy.
        dir_alpha (float): The alpha value for the Dirichlet distribution.

    Args:
        model (AlphaZeroModel): The AlphaZero model used for value and policy prediction.
        *args: Variable length argument list.
        dir_epsilon (float, optional): The epsilon value for adding Dirichlet noise to the prior policy. Defaults to 0.0.
        dir_alpha (float, optional): The alpha value for the Dirichlet distribution. Defaults to 0.3.
        **kwargs: Arbitrary keyword arguments.
    """

    model: AlphaZeroModel
    dir_epsilon: float
    dir_alpha: float

    def __init__(
        self, model: AlphaZeroModel, *args, dir_epsilon=0.0, dir_alpha=0.3, value_estimate = "nn", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.dir_epsilon = dir_epsilon
        self.dir_alpha = dir_alpha
        self.value_estimate = value_estimate
        self.ncols = self.model.observation_embedding.ncols

        if self.value_estimate == "perfect":
            desc = self.model.env.unwrapped.desc.tolist() # Get the original desc of the training env
            self.distances = fz_compute_distances(desc) # Compute the distances from each cell to the goal cell
            
    @th.no_grad()
    def value_function(self, node: Node) -> float:
        """
        Computes the value function for a given node.

        Args:
            node (Node): The node for which to compute the value function.

        Returns:
            float: The computed value function.
        """
        if node.is_terminal():
            return 0.0
        observation = node.observation
        # flatten the observation
        assert observation is not None
        # run the model
        # convert observation from int to tensor float 1x1 tensor
        assert node.env is not None

        value, policy = self.model.single_observation_forward(observation)
        # if root and dir_epsilon > 0.0, add dirichlet noise to the prior policy
        if node.parent is None and self.dir_epsilon > 0.0:
            if self.dir_alpha is not None:
                noise = th.distributions.dirichlet.Dirichlet(
                    th.ones_like(policy) * self.dir_alpha
                ).sample()
            else:
                # uniform distribution
                noise = th.ones_like(policy) / policy.numel()
            node.prior_policy = (
                1 - self.dir_epsilon
            ) * policy + self.dir_epsilon * noise
        else:
            node.prior_policy = policy

        return value if self.value_estimate == "nn" else fz_perfect_value(self.distances, node.observation, self.ncols, self.discount_factor)
