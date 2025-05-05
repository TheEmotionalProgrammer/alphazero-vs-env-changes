from collections import deque
import copy
from typing import Dict, List, Tuple
import gymnasium as gym
from gymnasium import Env
import numpy as np
from core.node import Node
import torch as th
from environments.observation_embeddings import CoordinateEmbedding, ObservationEmbedding
from policies.policies import Policy
from environments.lunarlander.lunar_lander import CustomLunarLander
import matplotlib.pyplot as plt
from core.utils import copy_environment, print_obs


class MCTS:

    """
    This class contains the basic MCTS algorithm without assumptions on the leafs value estimation.
    """

    root_selection_policy: Policy
    selection_policy: Policy

    def __init__(
        self,
        selection_policy: Policy,
        discount_factor: float = 1.0,
        root_selection_policy: Policy | None = None,
    ):
        if root_selection_policy is None:
            root_selection_policy = selection_policy
        self.root_selection_policy = root_selection_policy
        self.selection_policy = selection_policy  # the selection policy should return None if the input node should be expanded
        self.discount_factor = discount_factor

    def search(
        self,
        env: gym.Env,
        iterations: int,
        obs,
        reward: float,
    ) -> Node:
        
        """
        The main function of the MCTS algorithm. Returns the current root node with updated statistics.
        It builds a tree of nodes starting from the root, i.e. the current state of the environment.
        The tree is built iteratively, and the value of the nodes is updated as the tree is built.
        """
        
        assert isinstance(env.action_space, gym.spaces.Discrete) # Assert that the type of the action space is discrete

        new_env = copy_environment(env) # Copy the environment
        
        root_node = Node(
            env= new_env,
            parent=None,
            reward=reward,
            action_space=env.action_space,
            observation=obs,
            #ncols=self.ncols
        )

        root_node.value_evaluation = self.value_function(root_node) # Estimate the value of the root node

        self.backup(root_node, root_node.value_evaluation) # Update node statistics

        return self.build_tree(root_node, iterations)

    def build_tree(self, from_node: Node, iterations: int) -> Node:

        """
        Builds the tree starting from the input node.
        The tree is built iteratively, and the value of the nodes is updated as the tree is built.
        """

        while from_node.visits < iterations: # Fixed number of iterations

            selected_node_for_expansion, selected_action = self.traverse(from_node) # Traverse the existing tree until a leaf node is reached

            if selected_node_for_expansion.is_terminal(): # If the node is terminal, set its value to 0 and backup
                
                selected_node_for_expansion.value_evaluation = 0.0

                self.backup(selected_node_for_expansion, 0)

            else:

                eval_node = self.expand(selected_node_for_expansion, selected_action) # Expand the node
                value = self.value_function(eval_node) # Estimate the value of the node
                eval_node.value_evaluation = value # Set the value of the node
                self.backup(eval_node, value) # Backup the value of the node

        return from_node # Return the root node, which will now have updated statistics after the tree has been built

    def value_function(
        self,
        node: Node,
    ) -> float:
        
        """
        Depending on the specific implementation, the value of a node can be estimated in different ways.
        For this reason we leave the implementation of the value function to subclasses.
        In random rollout MCTS, the value is the sum of the future reward when acting with uniformly random policy.
        """
        
        return .0 

    def traverse(
        self, from_node: Node
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

        while not node.is_terminal():
            
            action = self.selection_policy.sample(node) # Select which node to step into

            if action not in node.children: # This means the node is not expanded, so we stop traversing the tree
                break

            node = node.step(action) # Step into the chosen node
            
        return node, action

    def expand(
        self, node: Node, action: int
    ) -> Node:
        
        """
        Expands the node and returns the expanded node.
        """

        # Copy the environment
        env = copy_environment(node.env)

        assert env is not None

        # Step into the environment

        observation, reward, terminated, truncated, _ = env.step(action)
        terminal = terminated

        assert not truncated
        if terminated:
            observation = None

        node_class = type(node)

        # Create the node for the new state
        new_child = node_class(
            env=env,
            parent=node,
            reward=reward,
            action_space=node.action_space,
            terminal=terminal,
            observation=observation,
            action=action,
            #ncols=self.ncols
        )

        node.children[action] = new_child # Add the new node to the children of the parent node

        return new_child

    def backup(self, start_node: Node, value: float, new_visits: int = 1) -> None:
        
        """
        Backups the value of the start node to its parent, grandparent, etc., all the way to the root node.
        Updates the statistic of the nodes in the path:
        - subtree_sum: the sum of the value of the node and its children
        - visits: the number of times the node has been visited
        - height: the maximum number of steps from the node to a leaf
        """

        node = start_node
        cumulative_reward = value

        while node is not None: # The parent is None if node is the root
            
            cumulative_reward *= self.discount_factor
            cumulative_reward += node.reward
            node.subtree_sum += cumulative_reward
            node.visits += new_visits

            # Update the height of the node
            if node.children:
                node.height = max(child.height for child in node.children.values()) + 1
            else:
                node.height = 0  # Leaf nodes have a height of 0

            # Reset the prior policy and value evaluation (mark as needing update)

            node.policy_value = None
            node.variance = None

            node = node.parent
    
class NoLoopsMCTS(MCTS):

    def __init__(self, reuse_tree, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.previous_root = None
        self.reuse_tree = reuse_tree

    def traverse(
        self, from_node: Node
    ) -> Tuple[Node, int]:
        
        """
        Traverses the tree starting from the input node until a leaf node is reached.
        Returns the node and action to be expanded next.
        Returns None if the node is terminal.
        Note: The selection policy returns None if the input node should be expanded.
        """

        visited = set()

        node = from_node

        visited.add(node.observation)

        action = self.root_selection_policy.sample(node, mask=node.mask) # Select which node to step into

        if action not in node.children: # If the selection policy returns None, this indicates that the current node should be expanded
            return node, action, visited
                        
        node = node.step(action)  # Step into the chosen node

        visited.add(node.observation)

        while not node.is_terminal():   

            action = self.selection_policy.sample(node, mask=node.mask)

            if action not in node.children: # This means the node is not expanded, so we stop traversing the tree
                break

            node = node.step(action) # Step into the chosen node

            visited.add(node.observation)

        return node, action, visited
    
    def expand(
        self, node: Node, action: int
    ) -> Node:
        
        """
        Expands the node and returns the expanded node.
        """

        # Copy the environment
        env = copy_environment(node.env)

        assert env is not None

        # Step into the environment

        observation, reward, terminated, truncated, _ = env.step(action)
        terminal = terminated

        assert not truncated
        if terminated:
            observation = None

        node_class = type(node)

        # Create the node for the new state
        new_child = node_class(
            env=env,
            parent=node,
            reward=reward,
            action_space=node.action_space,
            terminal=terminal,
            observation=observation,
            action=action,
        )

        node.children[action] = new_child # Add the new node to the children of the parent node

        return new_child
    
    def search(self, env: Env, iterations: int, obs, reward: float, lastaction: int = None) -> Node:

        if self.previous_root is None or not self.reuse_tree:
            
            root_node = Node(
                env = env,
                parent = None,
                reward = reward,
                action_space = env.action_space,
                observation = obs,
                terminal = False,
            )

            self.previous_root = root_node

            root_node.value_evaluation = self.value_function(root_node)
            self.backup(root_node, root_node.value_evaluation)

        else:

            root_node = self.previous_root
            
            found = False
            max_depth = 0
            for _, child in root_node.children.items():
                if child.observation == obs and child.height > max_depth:
                    found = True
                    max_depth = child.height
                    root_node = child
                    self.previous_root = root_node
            
            root_node.parent = None

            if not found:
                root_node = Node(
                    env = env,
                    parent = None,
                    reward = reward,
                    action_space = env.action_space,
                    observation = obs,
                    terminal = False,

                )

                self.previous_root = root_node

                root_node.value_evaluation = self.value_function(root_node)
                self.backup(root_node, root_node.value_evaluation)
                        
        counter = root_node.visits 
        
        while root_node.visits - counter < iterations:
            
            selected_node_for_expansion, selected_action, visited = self.traverse(root_node) # Traverse the existing tree until a leaf node is reached

            if selected_node_for_expansion.is_terminal(): # If the node is terminal, set its value to 0 and backup
                
                selected_node_for_expansion.value_evaluation = 0.0
                self.backup(selected_node_for_expansion, 0)

            else:
                
                eval_node = self.expand(selected_node_for_expansion, selected_action) # Expand the node
                
                value = self.value_function(eval_node) # Estimate the value of the node
                        
                eval_node.value_evaluation = value # Set the value of the node

                if eval_node.observation in visited:
                    eval_node.parent.mask[selected_action] = 0
                    eval_node.value_evaluation = 0.0
                    self.backup(eval_node, 0)
                else:
                    self.backup(eval_node, value) # If the parent has been masked, this will only update the visits

        return root_node # Return the root node,
    
    def backup(self, start_node: Node, value: float, new_visits: int = 1) -> None:
        
        """
        Backups the value of the start node to its parent, grandparent, etc., all the way to the root node.
        Updates the statistic of the nodes in the path:
        - subtree_sum: the sum of the value of the node and its children
        - visits: the number of times the node has been visited
        - height: the maximum number of steps from the node to a leaf
        """

        node = start_node
        cumulative_reward = value

        while node is not None: # The parent is None if node is the root
            
            cumulative_reward *= self.discount_factor
            cumulative_reward += node.reward
            node.subtree_sum += cumulative_reward
            node.visits += new_visits

            # Update the height of the node
            if node.children:
                node.height = max(child.height for child in node.children.values()) + 1
            else:
                node.height = 0  # Leaf nodes have a height of 0

            # Reset the prior policy and value evaluation (mark as needing update)
            node.policy_value = None
            node.variance = None

            node = node.parent
                
class RandomRolloutMCTS(MCTS):
    def __init__(self, rollout_budget=40, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_budget = rollout_budget

    def value_function(
        self,
        node: Node,
    ) -> float:
        
        """
        The standard value function for MCTS: 
        Sum of the future reward when acting with uniformly random policy.
        """

        # if the node is terminal, return 0
        if node.is_terminal():
            return 0.0

        # if the node is not terminal, simulate the enviroment with random actions and return the accumulated reward until termination
        accumulated_reward = 0.0
        discount = self.discount_factor
        env = copy_environment(node.env)
        assert env is not None
        for i in range(self.rollout_budget):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            #print(print_obs(env, obs))
            accumulated_reward += reward * (discount** (i+1))
            assert not truncated
            if terminated or truncated:
                break

        return accumulated_reward

class DistanceMCTS(MCTS):
    def __init__(self, embedding: CoordinateEmbedding, goal_state: int | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if goal_state is None:
            # the goal is max row max col
            goal_state = embedding.nrows * embedding.ncols - 1
        self.goal_state = goal_state
        self.embedding = embedding


    def value_function(
        self,
        node: Node,
    ) -> float:
        """
        The value function for MCTS is the distance to the goal state.
        """
        if node.is_terminal():
            return 0.0
        cols = self.embedding.ncols
        assert cols is not None
        rows = self.embedding.nrows
        assert rows is not None
        observation = node.observation
        assert observation is not None

        """
        There are (rows-1) x cols + 1 possible states. The player cannot be at the cliff, nor at the goal as the latter results in the end of the episode. What remains are all the positions of the first 3 rows plus the bottom-left cell.
        The observation is a value representing the player’s current position as current_row * nrows + current_col (where both the row and col start at 0).
        For example, the stating position can be calculated as follows: (rows-1) * cols + 0 = 36.
        """
        goal_row = self.goal_state // cols
        goal_col = self.goal_state % cols
        current_row = observation // cols
        current_col = observation % cols

        col_diff = abs(goal_col - current_col)
        row_diff = abs(goal_row - current_row)
        manhattan_distance = col_diff + row_diff
        # special case for cliffwalking env
        # if we are in the last row, we need to add two since we cannot go directly to the goal (cuz cliff)
        if cols == 12:
            if current_row == rows - 1:
                manhattan_distance += 2
            return - float(manhattan_distance)
        else:
            return self.discount_factor ** manhattan_distance


def compute_distances(lake_map: List[str]) -> Dict[int, int]:
    """
    Computes the distance from each cell to the goal cell, taking holes into account.
    """
    rows = len(lake_map)
    cols = len(lake_map[0])
    goal = (rows - 1, cols - 1)
    distances = {goal: 0}
    visited = set(goal)
    queue = deque([goal])

    while queue:
        row, col = queue.popleft()
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols and (new_row, new_col) not in visited:
                if lake_map[new_row][new_col] == 'H':
                    continue
                visited.add((new_row, new_col))
                distances[new_row, new_col] = distances[row, col] + 1
                queue.append((new_row, new_col))
    return distances


class LakeDistanceMCTS(MCTS):

    def __init__(self, lake_map: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distances = compute_distances(lake_map)
        self.ncols = len(lake_map[0])

    def get_distance(self, observation: int) -> int:
        current_row = observation // self.ncols
        current_col = observation % self.ncols
        return self.distances.get((current_row, current_col), float('inf'))

    def get_value(self, observation: int) -> float:
        distance = self.get_distance(observation)
        return self.discount_factor ** distance if distance != float('inf') else 0.0

    def value_function(
            self,
            node: Node,
        ) -> float:
        if node.is_terminal():
            return 0.0
        observation = node.observation
        assert observation is not None
        return self.get_value(observation)


