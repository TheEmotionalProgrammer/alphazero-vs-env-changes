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
from core.utils import copy_environment, observations_equal, actions_dict, print_obs

class MiniTrees(AlphaZeroMCTS):

    def __init__(
            self,
            model: AlphaZeroModel,
            selection_policy: Policy,
            threshold: float = 0.1, 
            discount_factor: float = 0.95,
            dir_epsilon: float = 0.0,
            dir_alpha: float = 0.3,
            root_selection_policy: Policy | None = None,
            predictor: str = "current_value",
            value_search: bool = False,
            value_estimate: str = "nn",
            update_estimator: bool = False
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
        self.trajectory = [] # List of tuples (node, action) that represent the trajectory sampled by unrolling the prior policy during detection
        self.problem_idx = None # Index of the problematic node in the trajectory, i.e. first node whose value estimate is disregarded
        self.predictor = predictor # The predictor to use for the n-step prediction
        self.value_search = value_search # If True, the agent will use the value search 
        self.update_estimator = update_estimator # If True, the agent will update the value estimator when a better estimate is found

        self.checked_obs = [] # List of observations that have been checked for detached unroll
        self.last_action = None # Last action taken in the environment
        
    def unroll(
            self,
            env: gym.Env,
            last_action: int | None, 
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

        len_traj = len(self.trajectory)

        if len_traj >= 1 and self.problem_idx is not None:
            if observations_equal(obs, self.trajectory[0][0].observation):
                print("Reusing Trajectory: ")
                return 0
            elif len_traj > 1 and observations_equal(obs, self.trajectory[1][0].observation):
                print("Reusing Trajectory: ")
                start_idx = 1
                self.trajectory = self.trajectory[start_idx:] 
                self.trajectory[0][0].parent = None # Set the parent of the root node to None
                self.problem_idx -= start_idx
                return 0
        
        self.trajectory = []
        self.problem_idx = None
    
        root_node = Node(
            env=copy_environment(env),  # Use the utility function
            parent=None,
            reward=reward,
            action_space=env.action_space,
            observation=obs,
        )
        
        original_root_node = (
            None if original_env is None else 
            Node(
                env=copy_environment(original_env),  # Use the utility function
                parent=None,
                reward=0,
                action_space=original_env.action_space,
                observation=obs,
            )
        )

        node = root_node

        value_estimate = 0.0

        val = self.value_function(node) 
        policy = node.prior_policy

        node.value_evaluation = val
        node.prior_policy = policy

        i_pred = val

        num_calls = 0

        #print(f"Value estimate: {val}, Prediction: {i_pred}", "obs", f"({print_obs(env, node.observation)[0]}, {print_obs(env, node.observation)[1]})")

        for i in range(n):

            value_estimate = value_estimate + (self.discount_factor**i) * node.reward

            action = th.argmax(policy).item()

            self.trajectory.append((node, action))

            child_env = copy_environment(node.env) # Use the utility function to copy the environment

            observation, reward, terminated, _, _ = child_env.step(action)

            child_node = Node(
                env=child_env,
                parent=None,
                reward=reward,
                action_space=child_env.action_space,
                observation=observation,
                terminal=terminated,
            )

            node = child_node

            val = node.reward if node.is_terminal() else self.value_function(node)
            node.value_evaluation = 0 if node.is_terminal() else val

            policy = None if node.is_terminal() else node.prior_policy

            num_calls += 1

            i_est = value_estimate + (self.discount_factor**(i+1)) * val
            
            if self.predictor == "original_env":
                i_pred = (
                    self.n_step_prediction(None, i+1, original_root_node)
                )

            if self.predictor == "current_value" and self.update_estimator and i_est > i_pred:
                i_pred = i_est # We found a better estimate for the value of the node, assuming we are following the optimal policy

            #print(f"Value estimate: {i_est}, Prediction: {i_pred}", "obs", f"({print_obs(env, node.observation)[0]}, {print_obs(env, node.observation)[1]})")

            # Add a very small delta to avoid division by zero
            i_pred = i_pred + 1e-9
            i_est = i_est + 1e-9

            if i_est/i_pred < 1 - self.threshold:

                print("Problem!")

                print("Problem detected at step", i+1, "with value estimate", i_est, "and prediction", i_pred)

                # We compute the safe number of steps estimation with this formula:
                # t = taken_steps - log(1-threshold)/log(discount_factor)
                # NOTE: at i in the for loop, we have taken i+1 steps
                
                safe_index = i+1 - (np.log(1-self.threshold)/np.log(self.discount_factor))

                safe_index = floor(safe_index)

                safe_index = max(safe_index, 0)
                # This is the number of steps we can take without encountering the problem
                # In this example situation: ()-()-()-x-()... it would be 2. Note that this also 
                # corresponds to the last safe state in the trajectory (since indexing starts from 0).
           
                problem_index = min(safe_index + 1, len(self.trajectory)) # We add 1 to include the first problematic node in the trajectory

                if problem_index == len(self.trajectory):
                    self.trajectory.append((node, None))

                self.problem_idx = problem_index
                self.trajectory = self.trajectory[:problem_index+1] # +1 to include the problematic node

                return num_calls
            
            if node.is_terminal():
                print("Terminal node reached")
                break

        print("No problem")

        return num_calls
             
        
    def detached_unroll(self, env: gym.Env, n: int, obs, reward: float, init_val: float , init_pol: float , original_env: None | gym.Env):

        """
        Unroll the prior from an arbitrary node (not necessarily the root). 
        Used in the value search planning style to check if a node with a higher value estimate than the problematic node can be found.
        """

        if original_env is not None:
            original_env.unwrapped.s = env.unwrapped.s # Set the state of the original environment to the state of the current environment
            original_env.unwrapped.lastaction = None
        
        root_node = Node(
            env=copy_environment(env),  # Use the utility function
            parent=None,
            reward=reward,
            action_space=env.action_space,
            observation=obs,
        )

        original_root_node = (
            None if original_env is None else 
            Node(
                env=copy_environment(original_env),  # Use the utility function
                parent=None,
                reward=0,
                action_space=original_env.action_space,
                observation=obs,
            )
        )

        node = root_node

        value_estimate = 0.0

        val = init_val
        policy = init_pol

        node.value_evaluation = val
        node.prior_policy = policy

        i_pred = val

        num_calls = 0

        for i in range(n):
                
                value_estimate = value_estimate + (self.discount_factor**i) * node.reward
    
                action = th.argmax(policy).item()
    
                # Create a new node with the action taken, without linking it to the parent node
                
                child_env = copy_environment(node.env) # Use the utility function to copy the environment

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

                val = self.value_function(node) if not node.is_terminal() else node.reward
                policy = node.prior_policy if not node.is_terminal() else None

                num_calls += 1

                i_est = value_estimate + (self.discount_factor**(i+1)) * val
                
                if self.predictor == "original_env":
                    i_pred = (
                        self.n_step_prediction(None, i+1, original_root_node)
                    )

                elif self.predictor == "current_value" and self.update_estimator and i_est > i_pred:
                    i_pred = i_est # We found a better estimate for the value of the node, assuming we are following the optimal policy

                # Add a very small delta to avoid division by zero
    
                i_pred = i_pred + 1e-9
                i_est = i_est + 1e-9
    
                if i_est/i_pred < 1 - self.threshold:
                    return True, num_calls # If a problem is detected, return True
                
                if node.is_terminal():
                    break

        return False, num_calls # No problem detected in the unroll

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
        
    def search(self, env: Env, iterations: int, obs, reward: float, original_env: Env | None, n: float = 5, last_action = None) -> Node:

        net_planning = 0 # To keep track of how many nn calls are made during planning

        num_calls = self.unroll(env, last_action , n, obs, reward, original_env) # We always unroll the prior before planning in azdetection

        net_planning += num_calls

        if self.problem_idx is None: # If no problem was detected we don't need to plan since we'll just follow the prior
            node = self.trajectory[0][0]
            self.backup(node, node.value_evaluation)
            return node, net_planning
        
        safe_length = max(len(self.trajectory)-1, 1) # Excludes the problematic node, we don't want to plan from there

        for idx in range(safe_length):
            
            if self.value_search:
                taken_actions  = [action for node, action in self.trajectory[:idx]] # List of actions that have been taken so far when this is the root

            root_node = self.trajectory[idx][0] # The node we are planning from 

            self.backup(root_node, root_node.value_evaluation) # Note that the value estimate of the root node is already set in the unrolling

            counter = root_node.visits # Avoids immediate stopping when we are reusing an old trajectory

            individual_iterations = iterations//safe_length

            while root_node.visits - counter < max((individual_iterations), 1) and net_planning < iterations:
                                    
                candidate_actions = taken_actions.copy() if self.value_search else None # We reset the candidate actions to the ones takes so far

                selected_node_for_expansion, selected_action = self.traverse(root_node, candidate_actions)

                if self.value_search:
                    candidate_actions.append(selected_action)

                if selected_node_for_expansion.is_terminal(): # If the node is terminal, set its value to 0 and backup
                        
                    selected_node_for_expansion.value_evaluation = 0.0
                    self.backup(selected_node_for_expansion, 0)
                
                else:

                    eval_node = self.expand(selected_node_for_expansion, selected_action)
                    eval_node.value_evaluation = self.value_function(eval_node)
                    net_planning += 1

                    check_val = eval_node.reward if eval_node.is_terminal() else eval_node.value_evaluation

                    if (
                        self.value_search 
                        and net_planning <= iterations - n
                        and not observations_equal(eval_node.observation, self.trajectory[self.problem_idx-1][0].observation)
                        and check_val >= self.trajectory[self.problem_idx-1][0].value_evaluation
                        and all(not observations_equal(eval_node.observation, obs) for obs in self.checked_obs)
                    ):

                        if eval_node.is_terminal():
                            return candidate_actions, net_planning
                        
                        # We create copies of the envs to avoid any interference with the standard ongoing planning
                        eval_node_env = copy_environment(eval_node.env)
                        original_env_copy = copy_environment(original_env) if original_env is not None else None
                        obs = eval_node.observation
                        reward = eval_node.reward

                        problem, num_calls = self.detached_unroll(eval_node_env, n, obs, reward, eval_node.value_evaluation , eval_node.prior_policy ,original_env_copy)
                        
                        net_planning += num_calls          

                        if not problem:
                            self.problem_idx = None # The problem has been solved
                            return candidate_actions, net_planning
                        else:
                            self.checked_obs.append(eval_node.observation)
                                                        
                    self.backup(eval_node, eval_node.value_evaluation)
        
        return self.trajectory[0][0], net_planning # If the problem is not solved, return the root node


    def traverse(
        self, from_node: Node, actions = None
    ) -> Tuple[Node, int]:
        
        """
        Same as AZMCTS but includes the option to track the actions taken 
        and append them to the actions list in input.
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










