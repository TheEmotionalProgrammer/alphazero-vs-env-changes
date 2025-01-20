
import numpy as np
import torch as th

from core.node import Node
from policies.policies import PolicyDistribution
from policies.utility_functions import get_children_policy_values, get_children_visits, get_transformed_default_values, policy_value, get_children_subtree_depth, get_transformed_mcts_t_values

class SelectionPolicy(PolicyDistribution):

    """
    Generic parent class for selection policies in MCTS.

    Input:
    - temperature: when equal to 0, we take the argmax of the policy distribution. Otherwise we sample.

    """

    def __init__(self, *args, temperature: float = 0.0, **kwargs) -> None:
        # by default, we use argmax in selection
        super().__init__(*args, temperature=temperature, **kwargs)

class UCT(SelectionPolicy):

    """
    UCT selection policy for MCTS.
    No prior policy is used and the selection is based on the Q value of the children + an exploration term.

    Input:
    - c: parameter that determines how much we want to explore, i.e. deviate from the exploitatory Q value follow.

    """

    def __init__(self, c: float, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def Q(self, node: Node) -> th.Tensor:

        """
        Default value estimate of the children which divides 
        the total reward of the subtree by the number of visits
        """

        return get_transformed_default_values(node, self.value_transform) 

    def _probs(self, node: Node) -> th.Tensor:
        
        child_visits = get_children_visits(node)
        #print("child_visits", child_visits)

        # if any child_visit is 0, we return 1 for all children with 0 visits
        if th.any(child_visits == 0):
            return child_visits == 0

        return self.Q(node) + self.c * th.sqrt(th.log(th.tensor(node.visits)) / child_visits)

class PolicyUCT(UCT):

    """
    Uses the same formula as UCT, but the way we calculate Q is different.
    This is because the way we evaluate trees is different (from standard visitation counts)
    and it's therefore beneficial to use the related Q estimate instead of the default mean Q.

    Input:
    - policy: The tree evaluation policy, e.g. visit count or mvc
    - discount factor: Usual env gamma discount factor
     
    """

    def __init__(self, *args, policy: PolicyDistribution, discount_factor: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.discount_factor = discount_factor

    def Q(self, node: Node) -> float:
        return get_children_policy_values(node, self.policy, self.discount_factor, self.value_transform)


class VarianceSelectionPolicy(PolicyUCT):
    
    """
    Selection policy based on the variance of the policy value estimates.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _probs(self, node: Node) -> th.Tensor:

        variances = get_children_variances(node, self.policy, self.discount_factor)

        # if any child_visit is 0
        if th.any(variances == 1):
            # return 1 for all children with 0 visits
            return variances == 1

        return self.Q(node) + self.c * variances 


class T_UCT(UCT):
    """
    Selection policy based on subtree depth estimates and visitation counts.
    """

    def __init__(self, c: float, discount_factor: float = 1.0 , *args,  **kwargs):
        super().__init__(c, *args, **kwargs)
        self.discount_factor = discount_factor

    def Q(self, node: Node) -> th.Tensor:
        return get_transformed_mcts_t_values(node, self.discount_factor, self.value_transform)

    def _probs(self, node: Node) -> th.Tensor:

        child_visits = get_children_visits(node)
        child_subtree_depths = get_children_subtree_depth(node)

        # print("child_visits", child_visits)
        # print("child_subtree_depths", child_subtree_depths)

        # if any child_visit is 0
        if th.any(child_visits == 0):
            # return 1 for all children with 0 visits
            return child_visits == 0
        #print(child_subtree_depths)
        #print("child_est_visits", child_subtree_depths)
        return self.Q(node) + self.c * child_subtree_depths * th.sqrt(th.log(th.tensor(node.visits))) / (child_visits + 1)

class PUCT(UCT):

    """
    PUCT selection policy for MCTS.
    Uses prior policy to weight the exploration term.
    """

    def _probs(self, node: Node) -> th.Tensor:
        child_visits = get_children_visits(node)
        # if any child_visit is 0
        unvisited = child_visits == 0
        if th.any(unvisited):
            return node.prior_policy * unvisited

        return self.Q(node) + self.c * node.prior_policy * th.sqrt(th.tensor(node.visits)) / (child_visits + 1)


class PolicyPUCT(PolicyUCT, PUCT):

    """
    Uses the PUCT formula and the generic Q estimate of PolicyUCT.
    """

    pass


selection_dict_fn = lambda c, policy, discount, value_transform: {
    
    "UCT": UCT(c, temperature=0.0, value_transform=value_transform),
    "PUCT": PUCT(c, temperature=0.0, value_transform=value_transform),
    "VarUCT": VarianceSelectionPolicy(c, policy = policy, discount_factor=discount,temperature=0.0, value_transform=value_transform),
    "PolicyUCT": PolicyUCT(c, policy=policy, discount_factor=discount,temperature=0.0, value_transform=value_transform),
    "T_UCT": T_UCT(c, discount_factor=discount,temperature=0.0, value_transform=value_transform),
    "PolicyPUCT": PolicyPUCT(c, policy=policy, discount_factor=discount,temperature=0.0, value_transform=value_transform),
}
