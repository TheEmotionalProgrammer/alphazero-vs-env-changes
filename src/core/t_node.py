from collections import Counter
from typing import Dict, Generic, List, TypeVar, Optional, Any, Callable, Tuple
import gymnasium as gym
import numpy as np
import torch as th
from core.node import Node


ObservationType = TypeVar("ObservationType")

NodeType = TypeVar("NodeType", bound="Node")

class T_Node(Node):

    parent: Optional["Node[ObservationType]"]
    children: Dict[int, "Node[ObservationType]"]
    visits: int = 0
    subtree_sum: float = 0.0  # sum of reward and value of all children
    value_evaluation: float  # Expected future reward
    reward: float  # Reward received when stepping into this node
    action_space: gym.spaces.Discrete  
    observation: Optional[ObservationType]
    prior_policy: th.Tensor
    env: Optional[gym.Env]
    variance: float | None = None
    policy_value: float | None = None

    backup_visits: int = 0 # number of visits to backup, estimated by the original policy
    whatif_value: float = 0 # value of the node estimated by the original policy
    
    def __init__(
        self,
        env: gym.Env,
        parent: Optional["Node[ObservationType]"],
        reward: float,
        action_space: gym.spaces.Discrete,
        observation: Optional[ObservationType],
        terminal: bool = False,
    ):
                
        super().__init__(env, parent, reward, action_space, observation, terminal)
        self.subtree_depth = 0 if terminal else 1



            
    
        
