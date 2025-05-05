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
    def __init__(self, discount_factor, **kwargs):

        super().__init__(**kwargs)
        self.discount_factor = discount_factor

    def _probs(self, node: Node) -> th.Tensor:
        
        vals = get_children_policy_values(node, self, self.discount_factor, self.value_transform)
        vals = th.nan_to_num(vals)
        vals = th.exp(vals - vals.max())

        return vals
    
class PriorPolicy(PolicyDistribution):
    """
    Prior Evaluator. 
    The action is chosen based on the prior policy of the children nodes.
    """

    def _probs(self, node: Node) -> th.Tensor:
        return node.prior_policy
    
class PriorValuePolicy(PolicyDistribution):
    """
    Prior Evaluator. 
    The action is chosen based on the prior policy of the children nodes.
    """

    def _probs(self, node: Node) -> th.Tensor:
        values = []
        for action, child in node.children.items():
            values.append(child.value_evaluation)
        values = th.tensor(values, dtype=th.float32)
        values = th.nan_to_num(values)
    
        return values
    
    
class MinimalVarianceConstraintPolicyPrior(MinimalVarianceConstraintPolicy):

    def _probs(self, node):

        beta = self.get_beta(node)

        # We compute Q and 1/V[Q] of the children
        normalized_vals, _ = get_children_policy_values_and_inverse_variance(node, self, self.discount_factor, self.value_transform)
        
        # We handle nan values and compute the final mvc distribution
        logits = beta * th.nan_to_num(normalized_vals)

        probs = th.exp(logits - logits.max())/node.prior_policy

        return probs

tree_eval_dict = lambda param, discount, c=1.0, temperature=None, value_transform=IdentityValueTransform: {
    "visit": VisitationPolicy(temperature, value_transform=value_transform),
    "qt_max": ValuePolicy(discount_factor=discount, temperature=temperature, value_transform=value_transform),
    "mvc": MinimalVarianceConstraintPolicy(discount_factor=discount, beta=param, temperature=temperature, value_transform=value_transform),
    "mvc_prior": MinimalVarianceConstraintPolicyPrior(discount_factor=discount, beta=param, temperature=temperature, value_transform=value_transform),
    "prior": PriorPolicy(temperature, value_transform=value_transform),
    "prior_value": PriorValuePolicy(temperature, value_transform=value_transform),
}
