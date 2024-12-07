from ast import List, Tuple
import copy
from math import floor
import gym
from gymnasium import Env
import torch as th
from az.model import AlphaZeroModel
from az.azmcts import AlphaZeroMCTS
from core.node import Node
from policies.policies import Policy
import numpy as np


class AlphaZeroDetector(AlphaZeroMCTS):

    """
    The algorithm alternates these steps after each step in the real environment:
    1. Unroll the prior AZ policy for n steps.
    2. Detect if the value function has changed, suggesting environment changes.
    3. Grow the unrolled trajectory into a planning tree to overcome the changes.
    """

    def __init__(
            self,
            predictor,
            model: AlphaZeroModel,
            selection_policy: Policy,
            threshold: float = 0.9, 
            discount_factor: float =0.95
    ):
        super().__init__(
            model = model,
            selection_policy = selection_policy,
            discount_factor = discount_factor
        )

        self.threshold = threshold
        self.expanded_nodes = [] # List of nodes that have been expanded by either the search or unroll methods
        
    def unroll(
            self,
            env: gym.Env, 
            n: int,
            obs,
            reward: float,
            original_env: None | gym.Env = None,
    ) -> List[Tuple[Node, int]]:
        
        """
        Unroll the prior AZ policy for n steps. 
        At every step, the cumulative value estimate is compared to the prediction to detect any changes.
        The policy is always unrolled from the root node.
        When a change is detected, the trajectory until that change is returned.
        """

        trajectory = []

        assert isinstance(env.action_space, gym.spaces.Discrete) # Assert that the type of the action space is discrete
        assert n > 0

        root_node = Node(
            env=copy.deepcopy(env),
            parent=None,
            reward=reward,
            action_space=env.action_space,
            observation=obs,
        )

        original_root_node = (
            None if original_env is None else 
            Node(
                env=copy.deepcopy(original_env),
                parent=None,
                reward=reward,
                action_space=original_env.action_space,
                observation=obs,
            )
        )

        node = root_node


        value_estimate = 0.0

        for i in range(n):

            if node.is_terminal():
                break

            value_estimate = value_estimate + (self.discount_factor**i) * node.reward

            policy = self.model.single_observation_forward(node.observation)[1]

            action = th.argmax(policy).item()

            trajectory.append((node, action)) # We store the trajectory of transitions for later use

            if action not in node.children:

                node = self.expand(node, action)
                
            else:

                node = node.step(action)

            i_est = value_estimate + (self.discount_factor**(i+1)) * self.value_function(node)
            i_pred = self.n_step_prediction(node, i+1, original_root_node)

            if i_est/i_pred < 1 - self.threshold:

                problem_index = floor(i - (np.log(1-self.threshold)/np.log(self.discount_factor)))

                return trajectory[:problem_index] 


    def n_step_prediction(self, node: Node, n: int, original_node: None | Node) -> float:

        """
        Predict the value of the node n steps into the future, based on a predictor.
        """

        if original_node is None:
             
             """
             Use the given self.predictor to predict the value of the node n steps into the future.
             Note that we cannot step into the environment n times, since we do not have the original env.
             """

        else:

            """
            Use the original env in the original node to predict the value of the node n steps into the future.
            Note that this is just for testing the mechanism, since having the original env would mean that we 
            could just compare the different transitions to detect changes.
            """

            value_estimate = 0.0
            node = original_node

            for i in range(n-1):
                    
                    _ , policy = self.model.single_observation_forward(node.observation)
                    action = th.argmax(policy).item()
    
                    if action not in node.children:
    
                        node = self.expand(node, action)
                        
                    else:
                        node = node.children[action]
    
                    value_estimate = value_estimate + (self.discount_factor**i) * node.reward

                    if i == n-1:

                        value_estimate = value_estimate + (self.discount_factor**n) * self.value_function(node) 


            return value_estimate


    def expand_tree(self, env: Env, num_expansions: int, obs, reward: float):
        """
        Grow the unrolled trajectory into a planning tree
        """
        pass




