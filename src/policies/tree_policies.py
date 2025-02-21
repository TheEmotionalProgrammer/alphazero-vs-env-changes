import torch as th

from core.node import Node
from policies.policies import PolicyDistribution
from policies.utility_functions import get_children_policy_values, get_children_policy_values_and_inverse_variance, get_children_visits
from policies.value_transforms import IdentityValueTransform


class VisitationPolicy(PolicyDistribution):

    """
    Visitation Counts Evaluator. 
    The action is chosen based on the number of visits to the children nodes performed during planning.
    """
    
    def _probs(self, node: Node) -> th.Tensor:
        visits = get_children_visits(node)
        return visits
    
class MinimalVarianceConstraintPolicy(PolicyDistribution):

    """
    Selects the action with the highest inverse variance of the Q value.

    Input:
    - beta: Beta parameter in the mvc formula.
    - discount_factor: Usual env gamma discount factor.

    """

    def __init__(self, beta: float, discount_factor = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.discount_factor = discount_factor

    def get_beta(self, node: Node):
        return self.beta

    def _probs(self, node: Node) -> th.Tensor:

        beta = self.get_beta(node)

        # We compute Q and 1/V[Q] of the children
        normalized_vals, inv_vars = get_children_policy_values_and_inverse_variance(node, self, self.discount_factor, self.value_transform)

        # We handle nan values and compute the final mvc distribution
        logits = beta * th.nan_to_num(normalized_vals)
        probs = inv_vars * th.exp(logits - logits.max())

        return probs


class ValuePolicy(PolicyDistribution):
    def __init__(self, discount_factor = 1.0, **kwargs):

        super().__init__(**kwargs)
        self.discount_factor = discount_factor
        self.temperature = 0.0 # Do not modify

    """
    Deterministic policy that selects the action with the highest value estimate 
    obtained by the MVC policy, but without the variance constraint.
    """

    def _probs(self, node: Node) -> th.Tensor:
        
        mvc_temp = 0.0
        mvc = MinimalVarianceConstraintPolicy(beta = 10.0, discount_factor=self.discount_factor, temperature=mvc_temp, value_transform=IdentityValueTransform)

        vals = get_children_policy_values(node, mvc, self.discount_factor, self.value_transform)

        return vals
    

tree_eval_dict = lambda param, discount, c=1.0, temperature=None, value_transform=IdentityValueTransform: {
    "visit": VisitationPolicy(temperature, value_transform=value_transform),
    "qt_max": ValuePolicy(discount_factor=discount, temperature=temperature, value_transform=value_transform),
    "mvc": MinimalVarianceConstraintPolicy(discount_factor=discount, beta=param, temperature=temperature, value_transform=value_transform),
}
