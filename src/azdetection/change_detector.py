from typing import Tuple
import copy
from math import floor, inf
import gym
from gymnasium import Env
import torch as th
from az.model import AlphaZeroModel
from az.azmcts import AlphaZeroMCTS
from core.node import Node
from policies.policies import Policy
import numpy as np
from environments.frozenlake.frozen_lake import actions_dict

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
            threshold: float = 0.1, 
            discount_factor: float = 0.9,
            dir_epsilon: float = 0.0,
            dir_alpha: float = 0.3,
            root_selection_policy: Policy | None = None,
    ):
        super().__init__(
            model = model,
            selection_policy = selection_policy,
            discount_factor = discount_factor,
            dir_epsilon = dir_epsilon,
            dir_alpha = dir_alpha,
            root_selection_policy = root_selection_policy,
        )

        self.threshold = threshold
        self.trajectory = [] # List of nodes that have been expanded by either the search or unroll methods
        self.problem_idx = None
        self.planning_style = "value_search"
        
    def unroll(
            self,
            env: gym.Env, 
            n: int,
            obs,
            reward: float,
            original_env: None | gym.Env = None,
    ):
        
        """
        Unroll the prior AZ policy for n steps. 
        At every step, the cumulative value estimate is compared to the prediction to detect any changes.
        The policy is always unrolled from the current root node.
        """

        if len(self.trajectory) > 1 and self.trajectory[1][0].observation == obs and self.problem_idx is not None:
            print("Reusing trajectory.")
            self.trajectory = self.trajectory[1:]

            self.problem_idx -= 1
            return
        
        if len(self.trajectory) == 1 and self.trajectory[0][0].observation == obs and self.problem_idx is not None:
            print("Reusing root node.")

            return

        self.trajectory = []
        self.problem_idx = None
    
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

        node.value_evaluation = self.value_function(node)
        self.backup(node, node.value_evaluation)

        for i in range(n):

            value_estimate = value_estimate + (self.discount_factor**i) * node.reward

            policy = self.model.single_observation_forward(node.observation)[1]

            action = th.argmax(policy).item()

            self.trajectory.append((node, action))

            # Create a new node with the action taken, without linking it to the parent node
            if action not in node.children:
                child_env = copy.deepcopy(node.env)

                observation, reward, terminated, truncated, _ = child_env.step(action)

                child_node = Node(
                    env=child_env,
                    parent=None,
                    reward=reward,
                    action_space=child_env.action_space,
                    observation=observation,
                    terminal=terminated,
                )

            node = child_node

            if node.is_terminal():
                break

            i_est = value_estimate + (self.discount_factor**(i+1)) * self.model.single_observation_forward(node.observation)[0]
            
            i_pred = self.n_step_prediction(None, i+1, original_root_node)

            # Add a very small delta to avoid division by zero

            i_pred = i_pred + 1e-9
            i_est = i_est + 1e-9

            #create coordinate lambda function that maps the observation to a 2D position
            coords = lambda observ: (observ // 8, observ % 8)
            
            print(f"Value estimate: {i_est}, Prediction: {i_pred}", "obs", f"({coords(node.observation)[0]}, {coords(node.observation)[1]})")

            if i_est/i_pred < 1 - self.threshold:


                problem_index = floor(i + 2 - (np.log(1-self.threshold)/np.log(self.discount_factor)))

                problem_obs = self.trajectory[problem_index-1][0].observation

                print(f"Problem detected at state ({coords(problem_obs)[0]}, {coords(problem_obs)[1]}), taking action {actions_dict[action]}")

                self.problem_idx = problem_index
                self.trajectory = self.trajectory[:problem_index]

                print([(coords(node.observation)[0], coords(node.observation)[1], actions_dict[action]) for node, action in self.trajectory])

                return

        return 

    def n_step_prediction(self, node: Node | None, n: int, original_node: None | Node) -> float:

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
        
    def search(self, env: Env, iterations: int, obs, reward: float, original_env: Env | None, n: float = 5) -> Node:

        self.unroll(env, n, obs, reward, original_env)

        if self.problem_idx is None:
            
            return self.trajectory[0][0]
        
        safe_length = max(len(self.trajectory)-1, 1)
        self.trajectory[self.problem_idx-1][0].value_evaluation = self.model.single_observation_forward(self.trajectory[self.problem_idx-1][0].observation)[0]

        for node, _ in self.trajectory:
                
                # Each node in the least has to be treated as a root
                # Check is the node has the field value_evaluation
                if not hasattr(node, "value_evaluation"):
                    node.value_evaluation = self.value_function(node)
                    self.backup(node, node.value_evaluation)
                
                if not hasattr(node, "prior_policy"):
                    node.prior_policy = self.model.single_observation_forward(node.observation)[1]

                # for each possibe action
                for action in range(node.action_space.n):
                    # if the action has not been taken
                    if action not in node.children:
                        # expand the node
                        eval_node = self.expand(node, action)
                        # evaluate the node
                        eval_node.value_evaluation = self.value_function(eval_node)
                        # backup the value
                        self.backup(eval_node, eval_node.value_evaluation)

        if self.planning_style == "classic":
        
            for idx in range(safe_length):
                counter = self.trajectory[idx][0].visits
                while self.trajectory[idx][0].visits - counter < iterations//len(self.trajectory):

                    selected_node_for_expansion, selected_action = self.traverse(self.trajectory[idx][0], None)
                    eval_node = self.expand(selected_node_for_expansion, selected_action)
                    eval_node.value_evaluation = self.value_function(eval_node)
                    self.backup(eval_node, eval_node.value_evaluation)

        elif self.planning_style == "value_search":

            # Looks for a node with value estimate greather than the one obtained by taking the problematic action from the problematic node
            # i.e. a value higher than the child self.trajectory[self.problem_idx][0] -> self.trajectory[self.problem_idx][1]
            # If it finds such a node, unrolls the trajectory from that node to check if it still encunters the problem
            # If it doesn't, the agent will take the sequence of actions that lead to that node in the real environment
            # If it does, the agent will keep searching for a node with a higher value estimate

            for idx in range(safe_length):

                counter = self.trajectory[idx][0].visits
                
                taken_actions  = [action for node, action in self.trajectory[:idx]] # List of actions that have been taken so far when this is the root

                while self.trajectory[idx][0].visits - counter < iterations//len(self.trajectory):

                    candidate_actions = taken_actions.copy()

                    selected_node_for_expansion, selected_action = self.traverse(self.trajectory[idx][0], candidate_actions)
                    candidate_actions.append(selected_action)
                    eval_node = self.expand(selected_node_for_expansion, selected_action)
                    eval_node.value_evaluation = self.value_function(eval_node)

                    if eval_node.value_evaluation > self.trajectory[self.problem_idx-1][0].value_evaluation:
                        
                        eval_node_env = copy.deepcopy(eval_node.env)
                        original_env_copy = copy.deepcopy(original_env)

                        if not self.detached_unroll(eval_node_env, n, obs, reward, original_env_copy): # If the unroll does not encounter the problem

                            self.problem_idx = None # The problem has been solved
                            return candidate_actions
                        else:
                            candidate_actions = taken_actions.copy()
                            
                    else: # If the value estimate is 
                        candidate_actions = taken_actions.copy()
                        
                    self.backup(eval_node, eval_node.value_evaluation)

        return self.trajectory[0][0] # If the problem is not solved, return the root node
    
    def detached_unroll(self, env: gym.Env, n: int, obs, reward: float, original_env: None | gym.Env) -> bool:

        original_env.unwrapped.s = env.unwrapped.s # Set the state of the original environment to the state of the current environment
        original_env.unwrapped.lastaction = None

        #check that the two observations are the same
        

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
                
                value_estimate = value_estimate + (self.discount_factor**i) * node.reward
    
                policy = self.model.single_observation_forward(node.observation)[1]
    
                action = th.argmax(policy).item()
    
                # Create a new node with the action taken, without linking it to the parent node
                
                child_env = copy.deepcopy(node.env)

                observation, reward, terminated, truncated, _ = child_env.step(action)

                child_node = Node(
                    env=child_env,
                    parent=None,
                    reward=reward,
                    action_space=child_env.action_space,
                    observation=observation,
                    terminal=terminated,
                )
    
                node = child_node
    
                if node.is_terminal():
                    break

                coords = lambda observ: (observ // 8, observ % 8)
    
                i_est = value_estimate + (self.discount_factor**(i+1)) * self.model.single_observation_forward(node.observation)[0]
                
                i_pred = self.n_step_prediction(None, i+1, original_root_node)
    
                # Add a very small delta to avoid division by zero
    
                i_pred = i_pred + 1e-9
                i_est = i_est + 1e-9

                #print(f"Value estimate: {i_est}, Prediction: {i_pred}", "obs", f"({coords(node.observation)[0]}, {coords(node.observation)[1]}), pred_obs")
    
                if i_est/i_pred < 1 - self.threshold:
                    #print("Problem detected in detached unroll")
                    return True # If a problem is detected, return True

        print("Clear detached unroll")

        return False # No problem detected in the unroll

            
    def traverse(
        self, from_node: Node, actions = None
    ) -> Tuple[Node, int]:
        
        """
        Traverses the tree starting from the input node until a leaf node is reached.
        Returns the node and action to be expanded next.
        Returns None if the node is terminal.
        Note: The selection policy returns None if the input node should be expanded.
        """

        node = from_node

        action = self.root_selection_policy.sample(node) # Select which node to step into
        
        if action not in node.children: # If the selection policy returns None, this indicates that the current node should be expanded
            return node, action
        
        node = node.step(action)  # Step into the chosen node

        if actions is not None:
            actions.append(action)

        while not node.is_terminal():
            
            action = self.selection_policy.sample(node) # Select which node to step into

            if action not in node.children: # This means the node is not expanded, so we stop traversing the tree
                break

            if actions is not None:
                actions.append(action)

            node = node.step(action) # Step into the chosen node
            
        return node, action
    

    




