from typing import Tuple
import copy
from math import floor
import gymnasium as gym
from gymnasium import Env
import torch as th
from az.model import AlphaZeroModel
from az.azmcts import AlphaZeroMCTS
from core.node import Node
from policies.policies import Policy
import numpy as np
from environments.frozenlake.frozen_lake import actions_dict

from policies.utility_functions import policy_value

class Octopus(AlphaZeroMCTS):

    def __init__(
            self,
            model: AlphaZeroModel,
            selection_policy: Policy,
            threshold: float = 0.1, 
            discount_factor: float = 0.9,
            dir_epsilon: float = 0.0,
            dir_alpha: float = 0.3,
            root_selection_policy: Policy | None = None,
            predictor: str = "current_value",
            value_estimate: str = "nn",
            var_penalty: float = 1.0,
            value_penalty: float = 0.0,
            update_estimator: bool = False,
            policy_det_rule: bool = False,
            reuse_tree: bool = True
    ):
        super().__init__(
            model = model,
            selection_policy = selection_policy,
            discount_factor = discount_factor,
            dir_epsilon = dir_epsilon,
            dir_alpha = dir_alpha,
            root_selection_policy = root_selection_policy,
            value_estimate = value_estimate
        )

        self.threshold = 0 if predictor == "original_env" else threshold # If we use the original env predictor, we can set the threshold arbitrarily low
        self.problem_idx = None # Index of the problematic node in the trajectory, i.e. first node whose value estimate is disregarded
        self.last_value_estimate = None 
        self.predictor = predictor # The predictor to use for the n-step prediction

        self.previous_root = None
        self.update_estimator = update_estimator
        self.var_penalty = var_penalty
        self.value_penalty = value_penalty

        self.policy_det_rule = policy_det_rule
        self.check_policy = copy.deepcopy(selection_policy)
        self.check_policy.c = 0.0

        self.reuse_tree = reuse_tree

    def coords(self, observ):
        return (observ // self.ncols, observ % self.ncols) if observ is not None else None

    def n_step_prediction(self, node: Node | None, n: int, original_node: None | Node) -> float:

        """
        Predict the value of the node n steps into the future, based on a predictor.
        """

        if self.predictor == "current_value":
             
             """
             We just use the nn value at the current step as the n-step prediction.
             """

             return node.value_evaluation

        elif self.predictor == "original_env":

            assert original_node is not None, "Original node must be provided to use the original env as a predictor."

            """
            Use the original env in the original node to predict the value of the node n steps into the future.
            Note that this is just for testing the mechanism, since having the original env would mean that we 
            could just compare the different transitions to detect changes.
            """

            value_estimate = 0.0
            node = original_node

            for i in range(n):
                    
                if node.is_terminal():
                    break

                value_estimate = value_estimate + (self.discount_factor**i) * node.reward
                
                _ , policy = self.model.single_observation_forward(node.observation)
                action = th.argmax(policy).item()

                if action not in node.children:

                    node = self.expand(node, action)
                    
                else:
                    node = node.children[action]

                if i == n-1:

                    value_estimate = value_estimate + (self.discount_factor**(i+1)) * self.value_function(node)

            return value_estimate   

    def traverse(
        self, from_node: Node
    ) -> Tuple[Node, int]:
        
        """
        Same as AZMCTS but performs problem detection simultaneously.
        """

        nodes = []
        prior_ok = True

        i = 0
        node = from_node

        nodes.append(node)
        
        action = self.root_selection_policy.sample(node) # Select which node to step into

        cumulated_reward = node.reward

        i_pred = node.value_evaluation
        
        if action not in node.children: # If the selection policy returns None, this indicates that the current node should be expanded
            return node, action, cumulated_reward, i, i_pred, nodes, prior_ok
        
        if self.policy_det_rule and th.argmax(node.prior_policy).item() != action and self.check_policy.sample(node) != action:
            prior_ok = False
        
        node = node.step(action)  # Step into the chosen node

        while not node.is_terminal():

            nodes.append(node)

            i+=1

            if self.discount_factor**i * node.value_evaluation > i_pred:
                i_pred = self.discount_factor**i * node.value_evaluation

            cumulated_reward += (self.discount_factor**i) * node.reward
            
            action = self.selection_policy.sample(node) # Select which node to step into

            if self.policy_det_rule and th.argmax(node.prior_policy).item() != action and self.check_policy.sample(node) != action:
                prior_ok = False

            if action not in node.children: # This means the node is not expanded, so we stop traversing the tree
                break

            node = node.step(action) # Step into the chosen node

        return node, action, cumulated_reward, i, i_pred, nodes, prior_ok
    
    def search(self, env: Env, iterations: int, obs, reward: float) -> Node:

        if self.previous_root is None or not self.reuse_tree:
            
            root_node = Node(
                env = env,
                parent = None,
                reward = reward,
                action_space = env.action_space,
                observation = obs,
                terminal = False,
                ncols=self.ncols
            )

            self.previous_root = root_node

            root_node.value_evaluation = self.value_function(root_node)
            self.backup(root_node, root_node.value_evaluation)

        else:

            # Check if the current env state is equivalent to one of the children of the previous root
            # If it is, we can just use that node as the root node

            root_node = self.previous_root
            
            found = False

            for action in root_node.children:
                
                if root_node.children[action].observation == obs:

                    root_node = root_node.children[action]
                    self.previous_root = root_node
                    root_node.parent = None
                    found = True

                    break

            if not found:
                root_node = Node(
                    env = env,
                    parent = None,
                    reward = reward,
                    action_space = env.action_space,
                    observation = obs,
                    terminal = False,
                    ncols=self.ncols
                )

                self.previous_root = root_node

                root_node.value_evaluation = self.value_function(root_node)
                self.backup(root_node, root_node.value_evaluation)
                
        counter = root_node.visits 
        
        while root_node.visits - counter < iterations:
            
            selected_node_for_expansion, selected_action, cumulated_reward, i, i_pred, nodes, prior_ok = self.traverse(root_node) # Traverse the existing tree until a leaf node is reached

            predictor_rootval = i_pred + 1e-9

            if selected_node_for_expansion.is_terminal(): 
                i_root = cumulated_reward + self.discount_factor**(i) * selected_node_for_expansion.reward
            else:
                i_root  = cumulated_reward + (self.discount_factor**(i)) * selected_node_for_expansion.value_evaluation

            criterion = (i_root/predictor_rootval < 1 - self.threshold)

            safe_index = floor(i - (np.log(1 - self.threshold) / np.log(self.discount_factor)))

            safe_index = max(0, safe_index)

            prob_index = min(safe_index+1, len(nodes)-1)

            if prior_ok and criterion:
                
                prob_node = nodes[prob_index]
                prob_node.problematic = True
                prob_node.var_penalty = self.var_penalty
                prob_node.value_evaluation = max(0, prob_node.value_evaluation - self.value_penalty)

                # observations = [self.coords(node.observation) for node in obss]

                # print("Problem detected on trajectory:", observations, "i_root:", i_root, "i_pred:", predictor_rootval)
                # print("Problematic node:", self.coords(prob_node.observation), "idx:", prob_index)

                #print("Problem detected at node", self.coords(selected_node_for_expansion.observation), "i_root:", i_root, "i_pred:", predictor_rootval)
                # selected_node_for_expansion.problematic = True
                # selected_node_for_expansion.var_penalty = self.var_penalty
                # selected_node_for_expansion.value_evaluation  = max(0, selected_node_for_expansion.value_evaluation - self.value_penalty)                
                #print("Problem detected on trajectory:", obss, "i_root:", i_root, "i_pred:", predictor_rootval)

            # Predict the value of the node n steps into the future

            if selected_node_for_expansion.is_terminal(): # If the node is terminal, set its value to 0 and backup
                
                selected_node_for_expansion.value_evaluation = 0.0
                self.backup(selected_node_for_expansion, 0)

            else:

                eval_node = self.expand(selected_node_for_expansion, selected_action) # Expand the node

                value = self.value_function(eval_node) # Estimate the value of the node
    
                eval_node.value_evaluation = value # Set the value of the node

                self.backup(eval_node, value) # Backup the value of the node

        return root_node # Return the root node, which will now have updated statistics after the tree has been built

   
