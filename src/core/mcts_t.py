from collections import deque
import copy
from typing import Dict, List, Tuple
import gymnasium as gym
import numpy as np
from core.node import Node
import torch as th
from environments.observation_embeddings import CoordinateEmbedding, ObservationEmbedding
from policies.policies import Policy
from core.mcts import MCTS

class MCTS_T(MCTS):

    """
    Implements the MCTS-T algorithm, which incorporates the estimate of subtree depth into the selection policy.
    """
    
    root_selection_policy: Policy
    selection_policy: Policy
    estimate_policy: Policy  # In MCTS_T, we keep track of the visitation counts as we acted with normal UCT.

    def __init__(
        self,
        selection_policy: Policy, # Should be T_UCT
        discount_factor: float = 1,
        root_selection_policy: Policy | None = None,
        estimate_policy: Policy | None = None # Should be UCT
    ):
        
        super().__init__(selection_policy, discount_factor, root_selection_policy)
        if estimate_policy is not None:
            self.estimate_policy = estimate_policy
        else:
            self.estimate_policy = selection_policy

    def search(self, env: gym.Env, iterations: int, obs, reward: float) -> Node:
        
        assert isinstance(env.action_space, gym.spaces.Discrete) # Assert that the type of the action space is discrete

        root_node = Node(
            env=copy.deepcopy(env),
            parent=None,
            reward=reward,
            action_space=env.action_space,
            observation=obs,
            backup_visits=1
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
                #print("Terminal node reached")
                selected_node_for_expansion.value_evaluation = 0.0

                selected_node_for_expansion.subtree_depth = 0
                selected_node_for_expansion.whatif_value = 0

                self.backup(selected_node_for_expansion, 0)

            else:

                eval_node = self.expand(selected_node_for_expansion, selected_action) # Expand the node
                value = self.value_function(eval_node) # Estimate the value of the node
                eval_node.value_evaluation = value # Set the value of the node
                eval_node.whatif_value = value # When we expand a leaf, we have to set the default whatif value to the value of the node
                self.backup(eval_node, value) # Backup the value of the node

        return from_node # Return the root node, which will now have updated statistics after the tree has been built

    def backup(self, start_node: Node, value: float, new_visits: int = 1) -> None:
        
        """
        Same as MCTS but we also estimate the subtree depth
        """

        node = start_node
        cumulative_reward = value

        while node is not None: # The parent is None if node is the root
            cumulative_reward *= self.discount_factor
            cumulative_reward += node.reward
            node.subtree_sum += cumulative_reward
            node.visits += new_visits

            node.variance = None
            node.policy_value = None

            # Update the subtree depth
            
            sum_visits = 0
            weighted_depths = 0
            whatif_value = 0
            
            for action in range(node.action_space.n):
                if action in node.children:
                
                    sum_visits += node.children[action].visits 
                    weighted_depths += node.children[action].subtree_depth * node.children[action].visits
                    whatif_value += node.children[action].backup_visits * node.children[action].subtree_sum
                   
                else:

                    sum_visits += 1
                    weighted_depths += 1

                    # We don't have to update the whatif value of the node if the action is not in the children
            
            # If the node does not have any children, we set whatif value to value
                    

            node.subtree_depth = weighted_depths / sum_visits if not node.is_terminal() else 0

            #print("Here")

            if node.is_terminal():

                node.whatif_value = node.reward

            else:
                node.whatif_value = whatif_value/node.backup_visits
            #print(node.whatif_value)

            node = node.parent

            # Counter loops
            if node is not None and node.observation == start_node.observation:
                #print("Counter loop detected")
                start_node.subtree_depth = 0
                start_node.value_evaluation = 0

    def traverse(
        self, from_node: Node
    ) -> Tuple[Node, int]:
        
        """
        Same as MCTS but we have to keep track of the backup counts as if we acted with the estimate policy.
        """

        node = from_node

        action = self.root_selection_policy.sample(node) # Select which node to step into
        whatif_action = self.estimate_policy.sample(node)
        
        if action not in node.children: # If the selection policy returns None, this indicates that the current node should be expanded
            return node, action
        
        node = node.step(action)  # Step into the chosen node
        
        if whatif_action in node.children:
            whatif_node = node.step(whatif_action)
            whatif_node.backup_visits += 1

        while not node.is_terminal():
            
            action = self.selection_policy.sample(node) # Select which node to step into
            whatif_action = self.estimate_policy.sample(node)

            if action not in node.children: # This means the node is not expanded, so we stop traversing the tree
                break

            node = node.step(action) # Step into the chosen node

            if whatif_action in node.children:
                whatif_node = node.step(whatif_action)
                whatif_node.backup_visits += 1
            
        return node, action
        

    def expand(
        self, node: Node, action: int
    ) -> Node:
        
        """
        Expands the node and returns the expanded node.
        Note: The function will modify the environment and the input node.
        """

        # if len(node.children) == int(node.action_space.n) - 1: # If this is the last child to be expanded, we do not need to copy the environment
        #     env = node.env
        #     node.env = None
        # else:
        env = copy.deepcopy(node.env) 

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
            backup_visits=1
        )


        node.children[action] = new_child # Add the new node to the children of the parent node

        return new_child


