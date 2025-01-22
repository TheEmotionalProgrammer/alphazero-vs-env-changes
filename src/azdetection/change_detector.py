from typing import Tuple
import copy
from math import floor, ceil
import gymnasium as gym
from gymnasium import Env
import torch as th
from az.model import AlphaZeroModel
from az.azmcts import AlphaZeroMCTS
from core.mcts_t import MCTS_T
from core.node import Node
from policies.policies import Policy
from policies.selection_distributions import PolicyUCT
import numpy as np
from environments.frozenlake.frozen_lake import actions_dict

from policies.utility_functions import policy_value

class AlphaZeroDetector(AlphaZeroMCTS):

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
            planning_style: str = "value_search",
            value_search: bool = False,
            value_estimate: str = "nn",
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

        self.threshold = 0.001 if predictor == "original_env" else threshold # If we use the original env predictor, we can set the threshold arbitrarily low
        self.trajectory = [] # List of tuples (node, action) that represent the trajectory sampled by unrolling the prior policy during detection
        self.problem_idx = None # Index of the problematic node in the trajectory, i.e. first node whose value estimate is disregarded
        self.planning_style = planning_style # The planning style to use when the problem is detected
        self.predictor = predictor # The predictor to use for the n-step prediction
        self.value_search = value_search # If True, the agent will use the value search 
        
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

        #create coordinate lambda function that maps the observation to a 2D position
        coords = lambda observ: (observ // self.ncols, observ % self.ncols)

        len_traj = len(self.trajectory)

        if len_traj >= 1 and self.problem_idx is not None:
            if obs == self.trajectory[0][0].observation:
                print("Reusing Trajectory: ")
                if self.planning_style != "mini_trees":
                    self.trajectory[0][0].parent = None # We set the parent of the root node to None
                print([(self.trajectory[i][0].observation // 8, self.trajectory[i][0].observation % 8) for i in range(len(self.trajectory))])
                return
            elif len_traj > 1 and obs == self.trajectory[1][0].observation:
                print("Reusing Trajectory: ")
                start_idx = 1
                self.trajectory = self.trajectory[start_idx:] 
                self.problem_idx -= start_idx
                if self.planning_style != "mini_trees":
                    self.trajectory[0][0].parent = None
                print([(self.trajectory[i][0].observation // 8, self.trajectory[i][0].observation % 8) for i in range(len(self.trajectory))])
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
                reward=0,
                action_space=original_env.action_space,
                observation=obs,
            )
        )

        node = root_node

        value_estimate = 0.0

        if self.planning_style == "mini_trees":

            val = self.value_function(node)
            policy = node.prior_policy

            node.value_evaluation = val
            node.prior_policy = policy
        
        elif self.planning_style == "connected":
                
            node.value_evaluation = self.value_function(node)
            self.backup(node, node.value_evaluation)

            val, policy = node.value_evaluation, node.prior_policy

        elif self.planning_style == "q-directed":
            node.value_evaluation = self.value_function(node)
            # node.policy_value = th.tensor(node.value_evaluation)
            # node.visits += 1
            self.backup(node, node.value_evaluation)
            val, policy = node.value_evaluation, node.prior_policy

        print(f"Value estimate: {val}, Prediction: {val}", "obs", f"({coords(node.observation)[0]}, {coords(node.observation)[1]})")

        for i in range(n):

            value_estimate = value_estimate + (self.discount_factor**i) * node.reward

            action = th.argmax(policy).item()

            self.trajectory.append((node, action))

            if self.planning_style == "mini_trees":

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

            elif self.planning_style == "connected": # connected or q-directed

                eval_node = self.expand(node, action)
                eval_node.value_evaluation = self.value_function(eval_node)

                node = eval_node
                self.backup(node, node.value_evaluation)

                if node.is_terminal():
                    break

                val, policy = node.value_evaluation, node.prior_policy
            
            elif self.planning_style == "q-directed":

                eval_node = self.expand(node, action)
                eval_node.value_evaluation = self.value_function(eval_node)

                node = eval_node
                # node.visits += 1
                # node.policy_value = th.tensor(node.value_evaluation)
                self.backup(node, node.value_evaluation)

                if node.is_terminal():
                    break

                val, policy = node.value_evaluation, node.prior_policy


            i_est = value_estimate + (self.discount_factor**(i+1)) * val
            
            i_pred = (
                self.n_step_prediction(None, i+1, original_root_node) if self.predictor == "original_env" else
                self.n_step_prediction(root_node, i+1, None)
            )

            print(f"Value estimate: {i_est}, Prediction: {i_pred}", "obs", f"({coords(node.observation)[0]}, {coords(node.observation)[1]})")

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

                safe_index = max(safe_index, 0)
                # This is the number of steps we can take without encountering the problem
                # In this example situation: ()-()-()-x-()... it would be 2. Note that this also 
                # corresponds to the last safe state in the trajectory (since indexing starts from 0).
           
                problem_index = min(safe_index + 1, n-1) # We add 1 to include the first problematic node in the trajectory

                if problem_index == len(self.trajectory):
                    self.trajectory.append((node, None))

                problem_obs = self.trajectory[problem_index][0].observation # Observation of the first node whose value estimate is disregarded
                print(f"Problem detected at state ({coords(problem_obs)[0]}, {coords(problem_obs)[1]}), after {problem_index} steps ")

                self.problem_idx = problem_index
                self.trajectory = self.trajectory[:problem_index+1] # +1 to include the problematic node

                #self.trajectory[problem_index][0].reward = -1 # Set the reward of the problematic node to 0

                print("Trajectory:", [(coords(node.observation)[0], coords(node.observation)[1], None if action is None else actions_dict[action]) for node, action in self.trajectory])

                return
             
        
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

                #coords = lambda observ: (observ // self.ncols, observ % self.ncols)

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
        
    def search(self, env: Env, iterations: int, obs, reward: float, original_env: Env | None, n: float = 5) -> Node:

        self.unroll(env, n, obs, reward, original_env) # We always unroll the prior before planning in azdetection

        if self.problem_idx is None: # If no problem was detected we don't need to plan since we'll just follow the prior
            node = self.trajectory[0][0]
            node.value_evaluation = self.value_function(node)
            self.backup(node, node.value_evaluation)
            return node
        
        safe_length = max(len(self.trajectory)-1, 1) # Excludes the problematic node, we don't want to plan from there

        if self.value_search:
            # We initialize the value estimate of the problematic node
            self.trajectory[self.problem_idx][0].value_evaluation = self.value_function(self.trajectory[self.problem_idx][0])

        if self.planning_style == "mini_trees":

            """
            Looks for a node with value estimate greather than the one obtained by taking the problematic action from the problematic node
            i.e. a value higher than the child self.trajectory[self.problem_idx][0] -> self.trajectory[self.problem_idx][1]
            If it finds such a node, unrolls the trajectory from that node to check if it still encunters the problem
            If it doesn't, the agent will take the sequence of actions that lead to that node in the real environment
            If it does, the agent will keep searching for a node with a higher value estimate
            """
            start_val = 0

            for idx in range(safe_length):
                
                if self.value_search:
                    taken_actions  = [action for node, action in self.trajectory[:idx]] # List of actions that have been taken so far when this is the root

                root_node = self.trajectory[idx][0]

                root_node.value_evaluation = self.value_function(root_node)
                self.backup(root_node, root_node.value_evaluation)

                if idx == 0:
                    start_val = root_node.value_evaluation

                counter = root_node.visits # Avoids immediate stopping when we are reusing an old trajectory

                while root_node.visits - counter < max(iterations//safe_length, 1):
                                        
                    candidate_actions = taken_actions.copy() if self.value_search else None # We reset the candidate actions to the ones takes so far

                    selected_node_for_expansion, selected_action = self.traverse(root_node, candidate_actions)

                    if self.value_search:
                        candidate_actions.append(selected_action)

                    if selected_node_for_expansion.is_terminal(): # If the node is terminal, set its value to 0 and backup

                        if self.value_search and selected_node_for_expansion.reward >= self.trajectory[self.problem_idx][0].value_evaluation:
                        #if self.value_search and selected_node_for_expansion.reward > start_val: # Potentially better but slower
                            self.problem_idx = None
                            return candidate_actions
                            
                        selected_node_for_expansion.value_evaluation = 0.0
                        self.backup(selected_node_for_expansion, 0)
                    
                    else:

                        eval_node = self.expand(selected_node_for_expansion, selected_action)
                        eval_node.value_evaluation = self.value_function(eval_node)

                        if self.value_search and eval_node.value_evaluation >= self.trajectory[self.problem_idx][0].value_evaluation:
                        #if self.value_search and eval_node.value_evaluation > start_val: # Potentially better but slower

                            # We create copies of the envs to avoid any interference with the standard ongoing planning
                            eval_node_env = copy.deepcopy(eval_node.env)
                            original_env_copy = copy.deepcopy(original_env)
                            obs = eval_node.observation
                            reward = eval_node.reward

                            if (
                                not self.detached_unroll(eval_node_env, n, obs, reward, original_env_copy) 
                                #and self.trajectory[self.problem_idx][0].observation != eval_node.observation
                            ): # If the unroll does not encounter the problem

                                self.problem_idx = None # The problem has been solved
                                return candidate_actions
                                                          
                        self.backup(eval_node, eval_node.value_evaluation)
        
        elif self.planning_style == "connected":
            
            """
            Unlike mini_trees, here we want to keep the unrolled trajectory as the starting planning tree.
            Therefore, we connect the nodes of such trajectory before the obstacle.
            Then, we plan with the given planning budget by selecting the nodes with the largest value.
            This is achieved by using the PolicyUCT selection policy and setting the c parameter to zero.
            This method should be combined with the mvc tree evaluation policy.
            """

            # assert isinstance(self.selection_policy, PolicyUCT)
            # assert self.selection_policy.c == 0

            # Backup the value of the problematic node

            root_node = self.trajectory[0][0]
            print("Num expanded children of the root:", len(root_node.children))
            start_val = root_node.value_evaluation
            
            counter = root_node.visits

            while root_node.visits - counter < iterations:

                candidate_actions  = [] if self.value_search else None

                selected_node_for_expansion, selected_action = self.traverse(root_node, candidate_actions) # Traverse the existing tree until a leaf node is reached

                if self.value_search:
                    candidate_actions.append(selected_action)

                if selected_node_for_expansion.is_terminal(): # If the node is terminal, set its value to 0 and backup

                    #if self.value_search and selected_node_for_expansion.reward > start_val:
                    if self.value_search and selected_node_for_expansion.reward > self.trajectory[self.problem_idx][0].value_evaluation:
                        self.problem_idx = None
                        return candidate_actions
                    
                    selected_node_for_expansion.value_evaluation = 0.0
                    self.backup(selected_node_for_expansion, 0)

                else:

                    eval_node = self.expand(selected_node_for_expansion, selected_action) # Expand the node
                    value = self.value_function(eval_node) # Estimate the value of the node
                    eval_node.value_evaluation = value # Set the value of the node

                    #if self.value_search and eval_node.value_evaluation > start_val:
                    if self.value_search and eval_node.value_evaluation > self.trajectory[self.problem_idx][0].value_evaluation:
                            
                            eval_node_env = copy.deepcopy(eval_node.env)
                            original_env_copy = copy.deepcopy(original_env)
                            obs = eval_node.observation
                            reward = eval_node.reward

                            if (
                                not self.detached_unroll(eval_node_env, n, obs, reward, original_env_copy) 
                            ): # If the unroll does not encounter the problem

                                self.problem_idx = None # The problem has been solved
                                return candidate_actions
                                
                    self.backup(eval_node, value) # Backup the value of the node

            return root_node # Return the root node, which will now have updated statistics after the tree has been built

        elif self.planning_style == "q-directed":

            assert isinstance(self.selection_policy, PolicyUCT)
            assert self.selection_policy.c == 0

            """

            We start from the unrolled trajectory. Then, we decide which node to plan from based on its Q value estimate.
            We compute the Q value estimate using policy_value(). 

            """

            safe_nodes = self.trajectory[:safe_length] # We only consider the nodes before the problematic node

            for _ in range(iterations):

                #safe_nodes = sorted(safe_nodes, key=lambda x: x[0].policy_value, reverse=True)

                # We select the node with the highest Q value estimate. To do this, we need to check which
                # node of the trajectory has the highest Q value estimate. We start from the root node.
                max_val = -np.inf
                chosen_node = None

                if self.value_search:
                    taken_actions = []
                    candidate_actions  = None

                for node, action in safe_nodes:

                    if self.value_search:
                        taken_actions.append(action)

                    node.policy_value = policy_value(node, self.selection_policy.policy, self.discount_factor)
                    print(node.policy_value)

                    # if not node.is_fully_expanded():
                    #     chosen_node = node
                    #     candidate_actions = taken_actions[:-1] if self.value_search else None
                    #     break
                    
                    if node.policy_value > max_val:
                        max_val = node.policy_value
                        chosen_node = node
                        candidate_actions = taken_actions[:-1] if self.value_search else None

                print("Chosen node value:", chosen_node.policy_value)

                parent = chosen_node.parent
                chosen_node.parent = None

                #print("Chosen node:", f"({chosen_node.observation // 8}, {chosen_node.observation % 8})")

                # We traverse the tree from the chosen node
                selected_node_for_expansion, selected_action = self.traverse(chosen_node, candidate_actions)

                if self.value_search:
                    candidate_actions.append(selected_action)

                if selected_node_for_expansion.is_terminal(): # If the node is terminal, set its value to 0 and backup

                    if self.value_search and selected_node_for_expansion.reward > self.trajectory[self.problem_idx][0].value_evaluation:
                        self.problem_idx = None
                        return candidate_actions
                
                    selected_node_for_expansion.value_evaluation = 0.0
                    self.backup(selected_node_for_expansion, 0)

                else:
                        
                    eval_node = self.expand(selected_node_for_expansion, selected_action)
                    value = self.value_function(eval_node)
                    eval_node.value_evaluation = value

                    if self.value_search and eval_node.value_evaluation > self.trajectory[self.problem_idx][0].value_evaluation:
                                
                        eval_node_env = copy.deepcopy(eval_node.env)
                        original_env_copy = copy.deepcopy(original_env)
                        obs = eval_node.observation
                        reward = eval_node.reward

                        if (
                            not self.detached_unroll(eval_node_env, n, obs, reward, original_env_copy) 
                        ):
                                
                            self.problem_idx = None
                            return candidate_actions

                    self.backup(eval_node, value)
                
                chosen_node.parent = parent

            return safe_nodes[0][0] # Return the root node, which will now have updated statistics after the tree has been built

        else:

            raise NotImplementedError("Planning style not recognized.")


        return self.trajectory[0][0] # If the problem is not solved, return the root node


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
    
    
class AlphaZeroDetector_T(AlphaZeroDetector, MCTS_T):

    def __init__(self, model: AlphaZeroModel, selection_policy: Policy, threshold: float = 0.1, discount_factor: float = 0.9, dir_epsilon: float = 0, dir_alpha: float = 0.3, root_selection_policy: Policy | None = None, predictor: str = "current_value", planning_style: str = "value_search", value_search: bool = False, estimation_policy: Policy | None = None):

        # Initialize MCTS_T first 
        MCTS_T.__init__(
            self,
            selection_policy=selection_policy,
            discount_factor=discount_factor,
            root_selection_policy=root_selection_policy,
            estimate_policy=estimation_policy,
        )

        AlphaZeroDetector.__init__(
            self,
            model,
            selection_policy,
            threshold,
            discount_factor,
            dir_epsilon,
            dir_alpha,
            root_selection_policy,
            predictor, 
            planning_style, 
            value_search
        )


    def traverse(
        self, from_node: Node, actions = None
    ) -> Tuple[Node, int]:
        
        """
        Same as AZMCTS but includes the option to track the actions taken 
        and append them to the actions list in input.
        """

        node = from_node

        action = self.root_selection_policy.sample(node) # Select which node to step into
        whatif_action = self.estimate_policy.sample(node)

        if whatif_action in node.children:
            whatif_node = node.step(whatif_action)
            whatif_node.backup_visits += 1
        
        if action not in node.children: # If the selection policy returns None, this indicates that the current node should be expanded
            return node, action
        
        node = node.step(action)  # Step into the chosen node
        
        if actions is not None:
            actions.append(action)

        while not node.is_terminal():
            
            action = self.selection_policy.sample(node) # Select which node to step into
            whatif_action = self.estimate_policy.sample(node)

            if whatif_action in node.children:
                whatif_node = node.step(whatif_action)
                whatif_node.backup_visits += 1

            if action not in node.children: # This means the node is not expanded, so we stop traversing the tree
                break

            if actions is not None:
                actions.append(action)

            node = node.step(action) # Step into the chosen node
            
        return node, action
    
    def search(self, env: Env, iterations: int, obs, reward: float, original_env: Env | None, n: float = 5) -> Node:

        self.unroll(env, n, obs, reward, original_env) # We always unroll the prior before planning in azdetection

        if self.problem_idx is None: # If no problem was detected we don't need to plan since we'll just follow the prior
            node = self.trajectory[0][0]
            node.backup_visits = 1
            node.value_evaluation = self.value_function(node)
            self.backup(node, node.value_evaluation)
            return node
        
        safe_length = max(len(self.trajectory)-1, 1) # Excludes the problematic node, we don't want to plan from there

        if self.value_search:
            # We initialize the value estimate of the problematic node
            self.trajectory[self.problem_idx][0].value_evaluation = self.value_function(self.trajectory[self.problem_idx][0])

        if self.planning_style == "mini_trees":

            """
            Looks for a node with value estimate greather than the one obtained by taking the problematic action from the problematic node
            i.e. a value higher than the child self.trajectory[self.problem_idx][0] -> self.trajectory[self.problem_idx][1]
            If it finds such a node, unrolls the trajectory from that node to check if it still encunters the problem
            If it doesn't, the agent will take the sequence of actions that lead to that node in the real environment
            If it does, the agent will keep searching for a node with a higher value estimate
            """
            start_val = 0

            for idx in range(safe_length):
                
                if self.value_search:
                    taken_actions  = [action for node, action in self.trajectory[:idx]] # List of actions that have been taken so far when this is the root

                root_node = self.trajectory[idx][0]

                root_node.value_evaluation = self.value_function(root_node)
                root_node.backup_visits = 1
                self.backup(root_node, root_node.value_evaluation)

                if idx == 0:
                    start_val = root_node.value_evaluation

                counter = root_node.visits # Avoids immediate stopping when we are reusing an old trajectory

                while root_node.visits - counter < max(iterations//safe_length, 1):
                                        
                    candidate_actions = taken_actions.copy() if self.value_search else None # We reset the candidate actions to the ones takes so far

                    selected_node_for_expansion, selected_action = self.traverse(root_node, candidate_actions)

                    if self.value_search:
                        candidate_actions.append(selected_action)

                    if selected_node_for_expansion.is_terminal(): # If the node is terminal, set its value to 0 and backup

                        if self.value_search and selected_node_for_expansion.reward > self.trajectory[self.problem_idx][0].value_evaluation:
                        #if self.value_search and selected_node_for_expansion.reward > start_val: # Potentially better but slower
                            self.problem_idx = None
                            return candidate_actions
                            
                        selected_node_for_expansion.value_evaluation = 0.0
                        selected_node_for_expansion.subtree_depth = 0
                        selected_node_for_expansion.whatif_value = 0
                        self.backup(selected_node_for_expansion, 0)
                    
                    else:

                        eval_node = self.expand(selected_node_for_expansion, selected_action)
                        value = self.value_function(eval_node)

                        eval_node.value_evaluation = value # Set the value of the node
                        eval_node.whatif_value = value # When we expand a leaf, we have to set the default whatif value to the value of the node

                        if self.value_search and eval_node.value_evaluation > self.trajectory[self.problem_idx][0].value_evaluation:
                        #if self.value_search and eval_node.value_evaluation > start_val: # Potentially better but slower

                            # We create copies of the envs to avoid any interference with the standard ongoing planning
                            eval_node_env = copy.deepcopy(eval_node.env)
                            original_env_copy = copy.deepcopy(original_env)
                            obs = eval_node.observation
                            reward = eval_node.reward

                            if (
                                not self.detached_unroll(eval_node_env, n, obs, reward, original_env_copy) 
                                #and self.trajectory[self.problem_idx][0].observation != eval_node.observation
                            ): # If the unroll does not encounter the problem

                                self.problem_idx = None # The problem has been solved
                                return candidate_actions
                                                          
                        self.backup(eval_node, eval_node.value_evaluation)
        
        elif self.planning_style == "connected":
            
            """
            Unlike mini_trees, here we want to keep the unrolled trajectory as the starting planning tree.
            Therefore, we connect the nodes of such trajectory before the obstacle.
            Then, we plan with the given planning budget by selecting the nodes with the largest value.
            This is achieved by using the PolicyUCT selection policy and setting the c parameter to zero.
            This method should be combined with the mvc tree evaluation policy.
            """

            # assert isinstance(self.selection_policy, PolicyUCT)
            # assert self.selection_policy.c == 0

            # Backup the value of the problematic node

            root_node = self.trajectory[0][0]
            root_node.backup_visits = 1
            print("Num expanded children of the root:", len(root_node.children))
            start_val = root_node.value_evaluation
            
            counter = root_node.visits

            while root_node.visits - counter < iterations:

                candidate_actions  = [] if self.value_search else None

                selected_node_for_expansion, selected_action = self.traverse(root_node, candidate_actions) # Traverse the existing tree until a leaf node is reached

                if self.value_search:
                    candidate_actions.append(selected_action)

                if selected_node_for_expansion.is_terminal(): # If the node is terminal, set its value to 0 and backup

                    #if self.value_search and selected_node_for_expansion.reward > start_val:
                    if self.value_search and selected_node_for_expansion.reward > self.trajectory[self.problem_idx][0].value_evaluation:
                        self.problem_idx = None
                        return candidate_actions
                    
                    selected_node_for_expansion.value_evaluation = 0.0
                    selected_node_for_expansion.subtree_depth = 0
                    selected_node_for_expansion.value_evaluation = 0.0
                    self.backup(selected_node_for_expansion, 0)

                else:

                    eval_node = self.expand(selected_node_for_expansion, selected_action) # Expand the node
                    value = self.value_function(eval_node) # Estimate the value of the node
                    eval_node.value_evaluation = value # Set the value of the node
                    eval_node.whatif_value = value

                    #if self.value_search and eval_node.value_evaluation > start_val:
                    if self.value_search and eval_node.value_evaluation > self.trajectory[self.problem_idx][0].value_evaluation:
                            
                            eval_node_env = copy.deepcopy(eval_node.env)
                            original_env_copy = copy.deepcopy(original_env)
                            obs = eval_node.observation
                            reward = eval_node.reward

                            if (
                                not self.detached_unroll(eval_node_env, n, obs, reward, original_env_copy) 
                            ): # If the unroll does not encounter the problem

                                self.problem_idx = None # The problem has been solved
                                return candidate_actions
                                
                    self.backup(eval_node, value) # Backup the value of the node

            return root_node # Return the root node, which will now have updated statistics after the tree has been built

        elif self.planning_style == "q-directed":

            assert isinstance(self.selection_policy, PolicyUCT)
            assert self.selection_policy.c == 0

            """

            We start from the unrolled trajectory. Then, we decide which node to plan from based on its Q value estimate.
            We compute the Q value estimate using policy_value(). 

            """

            safe_nodes = self.trajectory[:safe_length] # We only consider the nodes before the problematic node

            for _ in range(iterations):

                #safe_nodes = sorted(safe_nodes, key=lambda x: x[0].policy_value, reverse=True)

                # We select the node with the highest Q value estimate. To do this, we need to check which
                # node of the trajectory has the highest Q value estimate. We start from the root node.
                max_val = -np.inf
                chosen_node = None

                if self.value_search:
                    taken_actions = []
                    candidate_actions  = None

                for node, action in safe_nodes:

                    if self.value_search:
                        taken_actions.append(action)

                    node.policy_value = policy_value(node, self.selection_policy.policy, self.discount_factor)
                    print(node.policy_value)

                    # if not node.is_fully_expanded():
                    #     chosen_node = node
                    #     candidate_actions = taken_actions[:-1] if self.value_search else None
                    #     break
                    
                    if node.policy_value > max_val:
                        max_val = node.policy_value
                        chosen_node = node
                        candidate_actions = taken_actions[:-1] if self.value_search else None

                print("Chosen node value:", chosen_node.policy_value)

                parent = chosen_node.parent
                chosen_node.parent = None

                #print("Chosen node:", f"({chosen_node.observation // 8}, {chosen_node.observation % 8})")

                # We traverse the tree from the chosen node
                selected_node_for_expansion, selected_action = self.traverse(chosen_node, candidate_actions)

                if self.value_search:
                    candidate_actions.append(selected_action)

                if selected_node_for_expansion.is_terminal(): # If the node is terminal, set its value to 0 and backup

                    if self.value_search and selected_node_for_expansion.reward > self.trajectory[self.problem_idx][0].value_evaluation:
                        self.problem_idx = None
                        return candidate_actions
                
                    selected_node_for_expansion.value_evaluation = 0.0
                    self.backup(selected_node_for_expansion, 0)

                else:
                        
                    eval_node = self.expand(selected_node_for_expansion, selected_action)
                    value = self.value_function(eval_node)
                    eval_node.value_evaluation = value

                    if self.value_search and eval_node.value_evaluation > self.trajectory[self.problem_idx][0].value_evaluation:
                                
                        eval_node_env = copy.deepcopy(eval_node.env)
                        original_env_copy = copy.deepcopy(original_env)
                        obs = eval_node.observation
                        reward = eval_node.reward

                        if (
                            not self.detached_unroll(eval_node_env, n, obs, reward, original_env_copy) 
                        ):
                                
                            self.problem_idx = None
                            return candidate_actions

                    self.backup(eval_node, value)
                
                chosen_node.parent = parent

            return safe_nodes[0][0] # Return the root node, which will now have updated statistics after the tree has been built

        else:

            raise NotImplementedError("Planning style not recognized.")


        return self.trajectory[0][0] # If the problem is not solved, return the root node
        



        




