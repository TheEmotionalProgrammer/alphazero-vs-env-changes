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
from core.utils import copy_environment, actions_dict, print_obs, observations_equal

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
            var_penalty: float = 2.0,
            value_penalty: float = 1.0
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
        self.problem_value = None
        self.var_penalty = var_penalty
        self.value_penalty = value_penalty
        self.checked_obs = [] # List of observations that have been checked for problems
        self.max_planning = 0

        self.n = None
        
    def accumulate_unroll(self, env: gym.Env, n: int, obs, reward: float, original_env: None | gym.Env, env_action = None):

        append_nodes = False
        nn_calls = 0

        if self.trajectory == []:
            print("Starting unrolling from scratch")
            # We have started unrolling from scratch, so we have to create the first node
            self.trajectory = []
            self.problem_idx = None
            start = 0
            self.root_idx = 0
        
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

            append_nodes = True
            node = self.trajectory[-1][0]
            val = node.value_evaluation
            policy = node.prior_policy

            self.root_idx +=1
            self.trajectory[self.root_idx][0].parent = None # We set the parent of the root node to None

            start = len(self.trajectory)-1

            root_estimate = self.trajectory[0][0].value_evaluation

            original_root_node = None if original_env is None else \
            Node(
                env=copy.deepcopy(original_env),
                parent=None,
                reward=0,
                action_space=original_env.action_space,
                observation=self.trajectory[0][0].observation,
            )

            if original_root_node is not None:
                original_root_node.env.unwrapped.s = self.trajectory[0][0].env.unwrapped.s # Set the state of the original environment to the state of the current environment

            value_estimate = self.last_value_estimate # should take the final computation of the last trajectory
     
        if self.trajectory != [] and self.problem_idx is not None:

            len_traj = len(self.trajectory)

            if obs == self.trajectory[self.root_idx][0].observation:
                print("Reusing Trajectory: ")
                self.trajectory[self.root_idx][0].parent = None
                return 0
            elif self.root_idx < len_traj-1 and obs == self.trajectory[self.root_idx+1][0].observation:
                self.root_idx += 1
                self.trajectory[self.root_idx][0].parent = None
                return 0
            else: # the new node is not in the trajectory because the agent changed direction
                self.stop_unrolling = True
                return 0
                
        new_end = start + n
        i_pred = root_estimate

        for i in range(start, new_end):

            value_estimate = value_estimate + (self.discount_factor**i) * node.reward

            self.last_value_estimate = value_estimate

            action = th.argmax(policy).item()

            if not append_nodes or i != start:
                self.trajectory.append((node, action))
                append_nodes = True

            if action in node.children:
                
                node = node.children[action]
            
            else:

                eval_node = self.expand(node, action)
                eval_node.value_evaluation = self.value_function(eval_node)
                nn_calls += 1

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
                #safe_index = i + 1 - (np.log(i_est/i_pred)/np.log(self.discount_factor))
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

                # self.trajectory[problem_index][0].problematic = True
                # self.trajectory[problem_index][0].var_penalty = self.var_penalty
                # self.trajectory[problem_index][0].value_evaluation = self.trajectory[problem_index][0].value_evaluation - self.value_penalty
                
                problem_obs = self.trajectory[problem_index][0].observation # Observation of the first node whose value estimate is disregarded
                print(f"Problem detected at state ({print_obs(node.env, problem_obs)}), after {problem_index} steps ")

                self.problem_idx = problem_index

                self.problem_value = self.trajectory[self.problem_idx-1][0].value_evaluation
                #self.trajectory[problem_index][0].value_evaluation = max(0, self.trajectory[problem_index][0].value_evaluation - self.value_penalty)
                #self.trajectory[problem_index][0].problematic = True
                #self.trajectory[problem_index][0].var_penalty = self.var_penalty
                
                self.trajectory = self.trajectory[:problem_index+1] # +1 to include the problematic node

                print("Trajectory:", [(print_obs(node.env, node.observation), None if action is None else actions_dict(node.env)[action]) for node, action in self.trajectory])

                return nn_calls
            
            if i == new_end-1:
                # append the last node to the trajectory
                action = th.argmax(policy).item()

                self.trajectory.append((node, action))
        
        return nn_calls
                

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

        #print("Clear detached unroll")
        #print("No problem detected in detached unroll")
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
        
    def search(self, env: Env, iterations: int, obs, reward: float, original_env: Env | None, n: float = 5, env_action = None) -> Node:
        self.n = n
        self.max_planning = iterations
        net_planning = 0

        if self.problem_idx is not None:
            temporary_root = Node(
                env=copy.deepcopy(env),
                parent=None,
                reward=reward,
                action_space=env.action_space,
                observation=obs,
            )

            temporary_root.value_evaluation = self.value_function(temporary_root)

            if (
                self.problem_value is not None
                and net_planning <= iterations - n
                and temporary_root.value_evaluation >= self.problem_value
                and not observations_equal(temporary_root.observation, self.trajectory[self.problem_idx-1][0].observation)
                and all(not observations_equal(temporary_root.observation, obs) for obs in self.checked_obs)
                ):
                
                prob, nn_calls = self.detached_unroll(copy.deepcopy(env), n, obs, reward, temporary_root.value_evaluation, temporary_root.prior_policy, original_env)
                net_planning += nn_calls
                if not prob:
                    print("Problem solved, resume unrolling")
                    self.stop_unrolling = False
                    self.trajectory[self.problem_idx][0].problematic = False
                    self.trajectory = [] 
                    print("Problem solved")
                    self.problem_idx = None
                else:
                    self.checked_obs.append(temporary_root.observation)
                
        if not self.stop_unrolling:
            net_planning += self.accumulate_unroll(env, n, obs, reward, original_env, env_action) 
            root_node = self.trajectory[self.root_idx][0] #if (self.root_idx==len(self.trajectory)-1 or self.problem_idx is not None) else self.trajectory[self.root_idx+1][0]
            root_node.parent = None # We don't want to backtrack to nodes that are already surpassed
        else:
            # If the agent always follows the prior, then this will make it stay on the trajectory 
            new_root = True
            if self.trajectory[self.root_idx][0].observation == obs:
                root_node = self.trajectory[self.root_idx][0]
                root_node.parent = None
                new_root = False
                self.trajectory[self.root_idx] = (root_node, None)
            else:
                max_visits = 0
                for _, child in self.trajectory[self.root_idx][0].children.items():
                    if child.observation == obs and child.visits > max_visits:
                        root_node = child
                        root_node.parent = None
                        new_root = False
                        self.trajectory[self.root_idx] = (root_node, None)
                        max_visits = child.visits
                    
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
                
        while net_planning < iterations:

            candidate_actions  = [] if self.value_search else None

            selected_node_for_expansion, selected_action, net_planning = self.traverse(root_node, candidate_actions, net_planning) # Traverse the existing tree until a leaf node is reached

            if selected_node_for_expansion is None:
                return selected_action, net_planning 
            if self.value_search:
                candidate_actions.append(selected_action)

            if selected_node_for_expansion.is_terminal(): # If the node is terminal, set its value to 0 and backup
                
                selected_node_for_expansion.value_evaluation = 0.0
                self.backup(selected_node_for_expansion, 0)
                net_planning += 1

            else:

                eval_node = self.expand(selected_node_for_expansion, selected_action) # Expand the node

                value = self.value_function(eval_node) # Estimate the value of the node
                eval_node.value_evaluation = value # Set the value of the node

                net_planning += 1

                check_val = eval_node.reward if eval_node.is_terminal() else eval_node.value_evaluation

                if (
                    self.value_search 
                    and self.problem_idx is not None
                    and net_planning <= iterations - n
                    and not observations_equal(eval_node.observation, self.trajectory[self.problem_idx-1][0].observation)
                    and check_val >= self.trajectory[self.problem_idx-1][0].value_evaluation
                    and all(not observations_equal(eval_node.observation, obs) for obs in self.checked_obs)
                ):

                        if eval_node.is_terminal():
                            return candidate_actions, net_planning

                        eval_node_env = copy.deepcopy(eval_node.env)
                        original_env_copy = copy.deepcopy(original_env)
                        obs = eval_node.observation
                        reward = eval_node.reward

                        problem, nn_calls = self.detached_unroll(eval_node_env, n, obs, reward, eval_node.value_evaluation , eval_node.prior_policy ,original_env_copy)  
                        net_planning += nn_calls

                        if (
                            not problem
                        ): # If the rollout does not encounter any problem
                            
                            if eval_node.is_terminal(): # The episode will terminate
                                return candidate_actions, net_planning
                        
                            self.problem_idx = None
                            self.stop_unrolling = False
                            self.trajectory = []
                            self.root_idx = 0

                            return candidate_actions, net_planning
                        else:
                            self.checked_obs.append(eval_node.observation)
                            
                self.backup(eval_node, value) # Backup the value of the node

        return self.trajectory[self.root_idx][0], net_planning # Return the root node, which will now have updated statistics after the tree has been built

    def traverse(
        self, from_node: Node, actions = None, net_planning: int = 0
    ) -> Tuple[Node, int]:
        
        """
        Same as AZMCTS but includes the option to track the actions taken 
        and append them to the actions list in input.
        """

        node = from_node

        action = self.root_selection_policy.sample(node, node.mask) # Select which node to step into
        
        if action not in node.children: # If the selection policy returns None, this indicates that the current node should be expanded
            return node, action, net_planning
        
        node = node.step(action)  # Step into the chosen node

        if actions is not None:
            actions.append(action)
        
        if (
            self.value_search
            and self.problem_idx is not None
            and net_planning <= self.max_planning - self.n
            and not observations_equal(node.observation, self.trajectory[self.problem_idx-1][0].observation)
            and node.value_evaluation >= self.trajectory[self.problem_idx-1][0].value_evaluation
            and all(not observations_equal(node.observation, obs) for obs in self.checked_obs)  
        ):

            if node.is_terminal():
                return None, actions, net_planning

            node_env = copy.deepcopy(node.env)
            #original_env_copy = copy.deepcopy(original_env)
            obs = node.observation
            reward = node.reward

            problem, nn_calls = self.detached_unroll(node_env, self.n, obs, reward, node.value_evaluation, node.prior_policy, None)
            net_planning += nn_calls

            if (
                not problem
            ): # If the unroll does not encounter the problem
                self.problem_idx = None
                self.stop_unrolling = False
                self.trajectory = []
                self.root_idx = 0
                return None, actions, net_planning
            else:
                self.checked_obs.append(node.observation)
            
        while not node.is_terminal():
            
            action = self.selection_policy.sample(node, node.mask) # Select which node to step into

            if action not in node.children: # This means the node is not expanded, so we stop traversing the tree
                break

            if actions is not None:
                actions.append(action)

            node = node.step(action) # Step into the chosen node

            if (
                self.value_search
                and self.problem_idx is not None
                and net_planning <= self.max_planning - self.n
                and not observations_equal(node.observation, self.trajectory[self.problem_idx-1][0].observation)
                and node.value_evaluation >= self.trajectory[self.problem_idx-1][0].value_evaluation
                and all(not observations_equal(node.observation, obs) for obs in self.checked_obs)
            ):
                if node.is_terminal():
                    return None, actions, net_planning
                
                node_env = copy.deepcopy(node.env)
                #original_env_copy = copy.deepcopy(original_env)
                obs = node.observation
                reward = node.reward

                problem, nn_calls = self.detached_unroll(node_env, self.n, obs, reward, node.value_evaluation, node.prior_policy, None)

                net_planning += nn_calls

                if (
                    not problem
                ): # If the unroll does not encounter the problem
                    
                    self.problem_idx = None
                    self.stop_unrolling = False
                    self.trajectory = []
                    self.root_idx = 0
                    
                    return None, actions, net_planning 
                else:
                    self.checked_obs.append(node.observation)
        return node, action, net_planning

