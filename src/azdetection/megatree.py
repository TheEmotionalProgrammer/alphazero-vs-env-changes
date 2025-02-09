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

class MegaTree(AlphaZeroMCTS):

    """
    The algorithm alternates these steps after each step in the real environment:
    1. Unroll the prior AZ policy for n steps.
    2. Detect if the value function has changed, suggesting environment changes.
    3. Grow the unrolled trajectory into a planning tree to overcome the changes, following a specified algorithm.
    """

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
            update_estimator: bool = False,
            value_search: bool = False,
            value_estimate: str = "nn",
            var_penalty: float = 1.0
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
        self.last_value_estimate = None 
        self.root_idx = None
        self.predictor = predictor # The predictor to use for the n-step prediction
        self.update_estimator = update_estimator 
        self.value_search = value_search # If True, the agent will use the value search 
        self.stop_unrolling = False
        #self.time_left = th.inf
        self.problem_value = None
        self.var_penalty = var_penalty

        self.coords = lambda observ: (observ // self.ncols, observ % self.ncols) if observ is not None else None
        
    def accumulate_unroll(self, env: gym.Env, n: int, obs, reward: float, original_env: None | gym.Env, env_action = None) -> None:

        ziopera = False

        if self.trajectory == []:
            
            # We have started unrolling from scratch, so we have to create the first node
            self.trajectory = []
            self.problem_idx = None
            start = 0
            self.root_idx = 0
            #self.time_left = th.inf
        
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
                    reward=0,
                    action_space=original_env.action_space,
                    observation=obs,
                )
            )

            node = root_node

            value_estimate = 0.0

            node.value_evaluation = self.value_function(node)
            self.backup(node, node.value_evaluation)

            val, policy = node.value_evaluation, node.prior_policy

            root_estimate = self.n_step_prediction(None, 0, original_root_node) if self.predictor == "original_env" else node.value_evaluation

        if self.trajectory != [] and self.problem_idx is None:

            
            # We have already started unrolling and have not encountered a problem yet.
            # We can use the previous trajectory to continue unrolling, starting from the last node.

            ziopera = True
            node = self.trajectory[-1][0]
            val = node.value_evaluation
            policy = node.prior_policy

            self.root_idx +=1
            self.trajectory[self.root_idx][0].parent = None # We set the parent of the root node to None

            start = len(self.trajectory)-1

            root_estimate = self.trajectory[0][0].value_evaluation

            original_root_node = Node(
                env=copy.deepcopy(original_env),
                parent=None,
                reward=0,
                action_space=original_env.action_space,
                observation=self.trajectory[0][0].observation,
            )

            original_root_node.env.unwrapped.s = self.trajectory[0][0].env.unwrapped.s # Set the state of the original environment to the state of the current environment

            value_estimate = self.last_value_estimate # should take the final computation of the last trajectory
     
        if self.trajectory != [] and self.problem_idx is not None:

            len_traj = len(self.trajectory)

            if len_traj >= 1:
                if obs == self.trajectory[self.root_idx][0].observation:
                    print("Reusing Trajectory: ")
                    self.trajectory[self.root_idx][0].parent = None
                    print([(self.trajectory[i][0].observation // self.ncols, self.trajectory[i][0].observation % self.ncols) for i in range(len(self.trajectory))])
                    return
                elif len_traj > 1 and obs == self.trajectory[self.root_idx+1][0].observation:
                    print("Reusing Trajectory: ")
                    self.root_idx += 1
                    self.trajectory[self.root_idx][0].parent = None
                    return
                else: # the new node is not in the trajectory because the agent changed direction
                    self.stop_unrolling = True
                    return
            
            # We have started unrolling from scratch, but we can use an existing 

            if env_action in self.trajectory[self.root_idx][0].children:
                root_node = self.trajectory[self.root_idx][0].children[env_action]
                root_node.parent = None
                print("Gabibbo")
            else:
                root_node = Node(
                    env=copy.deepcopy(env),
                    parent=None,
                    reward=reward,
                    action_space=env.action_space,
                    observation=obs,
                )

            self.trajectory = []
            self.problem_idx = None
            self.root_idx = 0
            start = 0
            
            original_root_node = (
                None if original_env is None else 
                Node(
                    env=copy.deepcopy(original_env),
                    parent=None,
                    reward=0,
                    action_space=original_env.action_space,
                    observation=obs,
                )
            )

            node = root_node

            value_estimate = 0.0
            
            node.value_evaluation = self.value_function(node)
            self.backup(node, node.value_evaluation)

            val, policy = node.value_evaluation, node.prior_policy

            root_estimate = self.n_step_prediction(None, 0, original_root_node) if self.predictor == "original_env" else node.value_evaluation

        print("Current Trajectory: ")
        print([(self.trajectory[i][0].observation // self.ncols, self.trajectory[i][0].observation % self.ncols) for i in range(len(self.trajectory))])
        
        new_end = start + n
        i_pred = root_estimate

        for i in range(start, new_end):

            value_estimate = value_estimate + (self.discount_factor**i) * node.reward

            self.last_value_estimate = value_estimate

            action = th.argmax(policy).item()

            if not ziopera or i != start:
                self.trajectory.append((node, action))
                ziopera = True

            if action in node.children:
                
                node = node.children[action]
            
            else:

                eval_node = self.expand(node, action)
                eval_node.value_evaluation = self.value_function(eval_node)

                node = eval_node
                self.backup(node, node.value_evaluation)

            if node.is_terminal():
                break

            val, policy = node.value_evaluation, node.prior_policy
            
            
            i_est = value_estimate + (self.discount_factor**(i+1)) * val
            
            if self.predictor == "original_env":
                i_pred = (
                    self.n_step_prediction(None, i+1, original_root_node)
                )

            if self.predictor == "current_value" and self.update_estimator and i_est > i_pred:
                i_pred = i_est # We found a better estimate for the value of the node, assuming we are following the optimal policy
            
            # Add a very small delta to avoid division by zero
            i_pred = i_pred + 1e-9
            i_est = i_est + 1e-9

            if i_est/i_pred < 1 - self.threshold:

                # We compute the safe number of steps estimation with this formula:
                # t = taken_steps - log(1-threshold)/log(discount_factor)
                # NOTE: at i in the for loop, we have taken i+1 steps
                safe_index = i+1 - (np.log(1-self.threshold)/np.log(self.discount_factor))

                # if self.predictor == "current_value": # Log Error correction 
                #     safe_index -= np.log(i+1)

                safe_index = floor(safe_index)

                safe_index = max(safe_index, self.root_idx)
                # This is the number of steps we can take without encountering the problem
                # In this example situation: ()-()-()-x-()... it would be 2. Note that this also 
                # corresponds to the last safe state in the trajectory (since indexing starts from 0).
        
                problem_index = min(safe_index + 1, len(self.trajectory)) # We add 1 to include the first problematic node in the trajectory

                if problem_index == len(self.trajectory):
                    self.trajectory.append((node, None))

                self.trajectory[problem_index][0].problematic = True
                self.trajectory[problem_index][0].var_penalty = self.var_penalty
                
                problem_obs = self.trajectory[problem_index][0].observation # Observation of the first node whose value estimate is disregarded
                print(f"Problem detected at state ({self.coords(problem_obs)[0]}, {self.coords(problem_obs)[1]}), after {problem_index} steps ")

                self.problem_idx = problem_index
                #self.time_left = (self.problem_idx - self.root_idx) + 4 # We add 4 to allow for some extra steps to solve the problem
                #print("Time left:", self.time_left)

                self.problem_value = self.trajectory[self.problem_idx][0].value_evaluation
                
                self.trajectory = self.trajectory[:problem_index+1] # +1 to include the problematic node

                print("Trajectory:", [(self.coords(node.observation)[0], self.coords(node.observation)[1], None if action is None else actions_dict[action]) for node, action in self.trajectory])

                return
            
            if i == new_end-1:
                # append the last node to the trajectory
                action = th.argmax(policy).item()
                self.trajectory.append((node, action))
                

    def detached_unroll(self, env: gym.Env, n: int, obs, reward: float, original_env: None | gym.Env) -> bool:

        """
        Unroll the prior from an arbitrary node (not necessarily the root). 
        Used in the value search planning style to check if a node with a higher value estimate than the problematic node can be found.
        """

        if original_env is not None:
            original_env.unwrapped.s = env.unwrapped.s # Set the state of the original environment to the state of the current environment
            original_env.unwrapped.lastaction = None
        
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

        for i in range(n):
                
                value_estimate = value_estimate + (self.discount_factor**i) * node.reward
    
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

                val = self.value_function(node) 
                policy = node.prior_policy

                i_est = value_estimate + (self.discount_factor**(i+1)) * val
                
                i_pred = (
                    self.n_step_prediction(None, i+1, original_root_node) if self.predictor == "original_env" else
                    self.n_step_prediction(root_node, i+1, None)
                )
    
                # Add a very small delta to avoid division by zero
    
                i_pred = i_pred + 1e-9
                i_est = i_est + 1e-9

                if i_est/i_pred < 1 - self.threshold:
                    return True # If a problem is detected, return True

        print("Clear detached unroll")
        return False # No problem detected in the unroll

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
        
    def search(self, env: Env, iterations: int, obs, reward: float, original_env: Env | None, n: float = 5, env_action = None) -> Node:

        if self.problem_idx is not None:
            temporary_root = Node(
                env=copy.deepcopy(env),
                parent=None,
                reward=reward,
                action_space=env.action_space,
                observation=obs,
            )

            temporary_root.value_evaluation = self.value_function(temporary_root)

            # if self.problem_value is not None and temporary_root.value_evaluation > self.problem_value:
            #     print("Problem solved, resume unrolling")
            #     self.stop_unrolling = False
            #     self.trajectory[self.problem_idx][0].problematic = False
            #     self.trajectory = []
            #     self.problem_idx = None

        if not self.stop_unrolling:

            self.accumulate_unroll(env, n, obs, reward, original_env, env_action) 
        
            safe_length = max(len(self.trajectory)-1, 1) if self.problem_idx is not None else len(self.trajectory) 

            if self.value_search and self.problem_idx is not None:
                # We initialize the value estimate of the problematic node
                self.trajectory[self.problem_idx][0].value_evaluation = self.value_function(self.trajectory[self.problem_idx][0])
        
        if self.problem_idx is not None:
            distance = len(self.trajectory)

        if not self.stop_unrolling:
            root_node = self.trajectory[self.root_idx][0] 
            root_node.parent = None # We don't want to backtrack to nodes that are already surpassed
        else:

            new_root = True

            if self.trajectory[self.root_idx][0].observation == obs:
                root_node = self.trajectory[self.root_idx][0]
                root_node.parent = None
                new_root = False
                self.trajectory[self.root_idx] = (root_node, None)
                print("Same state")
            else:
                for _, child in self.trajectory[self.root_idx][0].children.items():
                    if child.observation == obs:
                        root_node = child
                        root_node.parent = None
                        new_root = False
                        self.trajectory[self.root_idx] = (root_node, None)
                        print("Supremo Gabibbo")
                        break
                
            if new_root:
                root_node = Node(
                    env=copy.deepcopy(env),
                    parent=None,
                    reward=reward,
                    action_space=env.action_space,
                    observation=obs,
                )
                root_node.value_evaluation = self.value_function(root_node)
                self.backup(root_node, root_node.value_evaluation)
                
        start_val = root_node.value_evaluation

        counter = root_node.visits 

        diff = n if not self.stop_unrolling else 0
        
        while root_node.visits - counter + diff < iterations:

            candidate_actions  = [] if self.value_search else None

            selected_node_for_expansion, selected_action = self.traverse(root_node, candidate_actions) # Traverse the existing tree until a leaf node is reached

            if self.value_search:
                candidate_actions.append(selected_action)

            if selected_node_for_expansion.is_terminal(): # If the node is terminal, set its value to 0 and backup
                
                selected_node_for_expansion.value_evaluation = 0.0
                self.backup(selected_node_for_expansion, 0)

            else:

                eval_node = self.expand(selected_node_for_expansion, selected_action) # Expand the node
                value = self.value_function(eval_node) # Estimate the value of the node
                eval_node.value_evaluation = value # Set the value of the node

                #if self.value_search and eval_node.value_evaluation > start_val:
                if (
                    self.value_search 
                    and self.problem_idx is not None
                    and eval_node.value_evaluation >= self.trajectory[self.problem_idx][0].value_evaluation
                    and eval_node.value_evaluation > start_val
                    ):
    
                        eval_node_env = copy.deepcopy(eval_node.env)
                        original_env_copy = copy.deepcopy(original_env)
                        obs = eval_node.observation
                        reward = eval_node.reward

                        if (
                            not self.detached_unroll(eval_node_env, distance, obs, reward, original_env_copy) 
                        ): # If the unroll does not encounter the problem
                            print("HERE")
                            return candidate_actions
                            
                self.backup(eval_node, value) # Backup the value of the node

        return root_node # Return the root node, which will now have updated statistics after the tree has been built

        


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
    
    