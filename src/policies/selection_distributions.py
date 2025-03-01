
import torch as th

from core.node import Node
from policies.policies import PolicyDistribution
from policies.tree_policies import MinimalVarianceConstraintPolicy, Q_max
from policies.utility_functions import get_children_policy_values, get_children_visits, get_transformed_default_values, get_children_policy_values_and_inverse_variance, value_evaluation_variance, reward_variance, get_children_q_max_values
from policies.value_transforms import IdentityValueTransform

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
        self.policy = policy# MinimalVarianceConstraintPolicy(beta = 10.0, discount_factor=0.95, temperature=0.0, value_transform=IdentityValueTransform)
        self.discount_factor = discount_factor

    def Q(self, node: Node) -> float:
        
        return get_children_policy_values(node, self.policy, self.discount_factor, self.value_transform)
        #return get_children_q_max_values(node, self.discount_factor)
        #return get_transformed_default_values(node, self.value_transform)
    
class PolicyUCT_Var(UCT):

    def __init__(self, *args, policy, discount_factor: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.policy = policy
        self.discount_factor = discount_factor

    def Q(self, node: Node) -> float:
        return get_children_policy_values(node, self.policy, self.discount_factor, self.value_transform)

    def _probs(self, node: Node) -> th.Tensor:

        children_visits = get_children_visits(node)
        # if any child_visit is 0
        if th.any(children_visits == 0):
            # return 1 for all children with 0 visits
            return children_visits == 0

        Qs, inv_vars = get_children_policy_values_and_inverse_variance(node, self.policy, self.discount_factor, self.value_transform)
        children_vars = 1/inv_vars

        if isinstance(self.policy, th.distributions.Categorical):
            pi = self.policy
        else:
            pi = self.policy.softmaxed_distribution(node, include_self=True)

        probabilities_squared = pi.probs**2  # type: ignore
        own_propability_squared = probabilities_squared[-1]
        child_propabilities_squared = probabilities_squared[:-1]

        var_root = reward_variance(node) + self.discount_factor**2 * (
            own_propability_squared * value_evaluation_variance(node)
            + (child_propabilities_squared * children_vars).sum()
        )

        inv_variance = th.tensor(1/var_root)
        
        # if th.log(inv_variance) < 0: # Avoids taking the square root of a negative number
        #     inv_variance = th.tensor(1)
        
        return Qs + self.c * th.sqrt(th.log(inv_variance)) * th.sqrt(children_vars)

class PolicyMax(SelectionPolicy):
    
    """
    Selects the action with the highest value estimate obtained by the MVC policy, but without the variance constraint.
    """

    def __init__(self, discount_factor = 1.0, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.discount_factor = discount_factor

    def _probs(self, node: Node) -> th.Tensor:

        children_visits = get_children_visits(node)

        # if any child_visit is 0
        if th.any(children_visits == 0):
            # return 1 for all children with 0 visits
            return children_visits == 0
        
        vals = get_children_q_max_values(node, self.discount_factor)

        return vals

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
    
class PolicyPUCT_Var(PolicyUCT, PUCT):
    
    def __init__(self, *args, discount_factor: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.policy = MinimalVarianceConstraintPolicy(beta = 10.0, discount_factor=discount_factor, temperature=None, value_transform=IdentityValueTransform)
        self.discount_factor = discount_factor

    def _probs(self, node: Node) -> th.Tensor:

        children_visits = get_children_visits(node)
        # if any child_visit is 0
        if th.any(children_visits == 0):
            # return 1 for all children with 0 visits
            return children_visits == 0

        Qs, inv_vars = get_children_policy_values_and_inverse_variance(node, self.policy, self.discount_factor, self.value_transform)
        inv_variance = th.tensor(1/value_evaluation_variance(node))
            
        return Qs + self.c * node.prior_policy *  th.sqrt(inv_variance) / (1 +  th.sqrt(inv_vars))


selection_dict_fn = lambda c, policy, discount, value_transform: {
    "UCT": UCT(c, temperature=0.0, value_transform=value_transform),
    "PUCT": PUCT(c, temperature=0.0, value_transform=value_transform),
    "PolicyUCT": PolicyUCT(c, policy=policy, discount_factor=discount,temperature=0.0, value_transform=value_transform),
    "PolicyPUCT": PolicyPUCT(c, policy=policy, discount_factor=discount,temperature=0.0, value_transform=value_transform),
    "PolicyUCT_Var": PolicyUCT_Var(c, policy=policy, discount_factor=discount,temperature=0.0, value_transform=value_transform),
    "PolicyPUCT_Var": PolicyPUCT_Var(c, policy=policy, discount_factor=discount,temperature=0.0, value_transform=value_transform),
    "PolicyMax": PolicyMax(discount_factor=discount, temperature=0.0, value_transform=value_transform)
}
