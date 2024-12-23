
import numpy as np
import torch as th

from core.node import Node
from policies.policies import PolicyDistribution
from policies.utility_functions import get_children_policy_values, get_children_visits, get_transformed_default_values, policy_value, get_children_subtree_depth, get_transformed_mcts_t_values

# use distributional selection policies instead of OptionalPolicy
class SelectionPolicy(PolicyDistribution):
    def __init__(self, *args, temperature: float = 0.0, **kwargs) -> None:
        # by default, we use argmax in selection
        super().__init__(*args, temperature=temperature, **kwargs)


class UCT(SelectionPolicy):

    """
    UCT selection policy for MCTS.
    No prior policy is used and the selection is based on the Q value of the children + an exploration term.
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
        # if any child_visit is 0
        if th.any(child_visits == 0):
            # return 1 for all children with 0 visits
            return child_visits == 0

        return self.Q(node) + self.c * th.sqrt(th.log(th.tensor(node.visits)) / child_visits)

class PolicyUCT(UCT):

    """
    Aka MVC-UCT.
    Used the same formula as UCT, but the way we calculate Q is different.
    This is because the way we evaluate trees is different (from standard visitation counts)
    and it's therefore beneficial to use the related Q estimate instead of the default mean Q.
    """

    def __init__(self, *args, policy: PolicyDistribution, discount_factor: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.discount_factor = discount_factor

    def Q(self, node: Node) -> float:
        return get_children_policy_values(node, self.policy, self.discount_factor, self.value_transform)

class T_UCT(PolicyUCT):
    """
    Selection policy based on subtree depth estimates and visitation counts.
    """

    def __init__(self, c: float, *args,  **kwargs):
        super().__init__(c, *args, **kwargs)

    def _probs(self, node: Node) -> th.Tensor:

        child_visits = get_children_visits(node)
        child_subtree_depths = get_children_subtree_depth(node)

        # if any child_visit is 0
        if th.any(child_visits == 0):
            # return 1 for all children with 0 visits
            return child_visits == 0
        #print(child_subtree_depths)
        #print("child_est_visits", child_subtree_depths)
        return self.Q(node) + self.c * child_subtree_depths * th.sqrt(th.tensor(node.visits)) / (child_visits + 1)

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
    Aka MVC-PUCT.
    Uses the PUCT formula and the MVC Q estimate of PolicyUCT.
    """

    pass


selection_dict_fn = lambda c, policy, discount, value_transform: {
    "UCT": UCT(c, temperature=0.0, value_transform=value_transform),
    "PUCT": PUCT(c, temperature=0.0, value_transform=value_transform),
    "PolicyUCT": PolicyUCT(c, policy=policy, discount_factor=discount,temperature=0.0, value_transform=value_transform),
    "T_UCT": T_UCT(c, policy=policy, discount_factor=discount,temperature=0.0, value_transform=value_transform),
    "PolicyPUCT": PolicyPUCT(c, policy=policy, discount_factor=discount,temperature=0.0, value_transform=value_transform),
}
