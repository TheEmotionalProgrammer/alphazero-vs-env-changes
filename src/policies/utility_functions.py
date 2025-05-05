import torch as th
from core.node import Node
from policies.policies import PolicyDistribution
from policies.value_transforms import IdentityValueTransform, ValueTransform


def policy_value(
    node: Node,
    policy: PolicyDistribution | th.distributions.Categorical, 
    discount_factor: float,
):
    """
    Computes the Q estimates yielded by the evaluation policy 
    that we are currently using. For example, the default visit
    count evaluator induces the arithmetic mean Q estimate.
    """

    # If the node is terminal, its value is just the reward.
    if node.terminal:
        val = th.tensor(node.reward, dtype=th.float32)
        node.policy_value = val
        return val

    # If the value has already been computed, return it.
    if node.policy_value is not None:
        return node.policy_value
    
    # We compute pi(a|s) for all a, including the special action. 
    # This is done using the evaluation policy (e.g., MVC).
    # Note that the special action prob is appended after the evaluation policy logits.
    if isinstance(policy, th.distributions.Categorical): # No need to softmax again
        pi = policy
    else:
        pi = policy.softmaxed_distribution(node, include_self=True, action_mask=node.mask)

    probabilities: th.Tensor = pi.probs

    assert probabilities.shape[-1] == int(node.action_space.n) + 1

    own_propability = probabilities[-1]  
    child_propabilities = probabilities[:-1] 

    child_values = th.zeros_like(child_propabilities, dtype=th.float32) # For all the unexpanded children, the value is 0
    
    # We recursively compute the value estimates of the children 
    for action, child in node.children.items():
        child_values[action] = policy_value(child, policy, discount_factor) 
   
    # Finally, we compute the value estimate of the node
    val = node.reward + discount_factor * (
        own_propability * node.value_evaluation
        + (child_propabilities * child_values).sum()
    )

    node.policy_value = val

    return val

def policy_value_variance(
    node: Node,
    policy: PolicyDistribution | th.distributions.Categorical,
    discount_factor: float,
):
    """
    Computes the variance of the Q estimates 
    yielded by the evaluation policy that we are currently using.
    """
    
    # If this is already computed, return it
    if node.variance is not None:
        return node.variance
    
    # We compute pi(a|s) for all a, including the special action. 
    # This is done using the evaluation policy (e.g., MVC).
    # Note that the special action prob is appended after the evaluation policy logits.
    # Finally, everything is normalized to sum to 1.
    if isinstance(policy, th.distributions.Categorical):
        pi = policy
    else:
        pi = policy.softmaxed_distribution(node, include_self=True, action_mask=node.mask)

    probabilities_squared = pi.probs**2  
    own_propability_squared = probabilities_squared[-1]
    child_propabilities_squared = probabilities_squared[:-1]

    child_variances = th.zeros_like(child_propabilities_squared, dtype=th.float32)

    for action, child in node.children.items():
        child_variances[action] = policy_value_variance(
            child, policy, discount_factor
        )

    var = reward_variance(node) + discount_factor**2 * (
        own_propability_squared * value_evaluation_variance(node)
        + (child_propabilities_squared * child_variances).sum()
    )

    node.variance = var
    
    return var

def reward_variance(node: Node):

    """
    The variance of the reward of the node, 
    under the assumption that it is deterministic.
    """

    return 0.0

def value_evaluation_variance(node: Node):

    """
    The variance of the value estimate of the node.
    We set it to the hyperparameter var.
    """

    var = 1.0

    if node.terminal:
        return var / float(node.visits)

    return var
    
def get_children_policy_values(
    parent: Node,
    policy: PolicyDistribution,
    discount_factor: float,
    transform: ValueTransform = IdentityValueTransform,
) -> th.Tensor:

    vals = th.ones(int(parent.action_space.n), dtype=th.float32) * -th.inf

    for action, child in parent.children.items():
        vals[action] = policy_value(child, policy, discount_factor)

    vals = transform.normalize(vals)

    return vals
        
def get_children_inverse_variances(
    parent: Node, policy: PolicyDistribution, discount_factor: float
) -> th.Tensor:
    
    inverse_variances = th.zeros(int(parent.action_space.n), dtype=th.float32)
    
    for action, child in parent.children.items():
        inverse_variances[action] = 1.0 / policy_value_variance(
            child, policy, discount_factor
        )

    return inverse_variances

def get_children_variances(
    parent: Node, policy: PolicyDistribution, discount_factor: float
) -> th.Tensor:
    
    variances = th.ones(int(parent.action_space.n), dtype=th.float32)

    for action, child in parent.children.items():
        variances[action] = policy_value_variance(
            child, policy, discount_factor
        )

    return variances


def get_children_policy_values_and_inverse_variance(
    parent: Node,
    policy: PolicyDistribution,
    discount_factor: float,
    transform: ValueTransform = IdentityValueTransform,
    include_self: bool = False,
) -> tuple[th.Tensor, th.Tensor]:
    
    """
    Equivalent to calling get_children_policy_values and get_children_variances separately.
    Returns the normalized values and the inverse variances of the children.
    """

    # Values are initialized to -inf, inverse variances to 0 (so default var would be inf)
    vals = th.ones(int(parent.action_space.n) + include_self, dtype=th.float32) * -th.inf 
    inv_vars = th.zeros_like(vals + include_self, dtype=th.float32)

    for action, child in parent.children.items():

        vals[action] = policy_value(child, policy, discount_factor)
        inv_vars[action] = 1 / policy_value_variance(
            child, policy, discount_factor
        )

    # Add value and variance of the special action a_v
    if include_self:
        vals[-1] = parent.value_evaluation
        inv_vars[-1] = 1 / value_evaluation_variance(parent)

    normalized_vals = transform.normalize(vals)

    return normalized_vals, inv_vars

def get_children_visits(node: Node) -> th.Tensor:

    """
    Returns a tensor with the the number of visits of the children of the node.
    """

    visits = th.zeros(int(node.action_space.n), dtype=th.float32)
    for action, child in node.children.items():
        visits[action] = child.visits

    return visits

def get_transformed_default_values(node: Node, transform: ValueTransform = IdentityValueTransform) -> th.Tensor:

    """
    Returns the value estimates of the children of the node.
    The default estimate is used, which is the total reward of the subtree divided by the number of visits.
    """

    vals = th.ones(int(node.action_space.n), dtype=th.float32) * -th.inf 

    for action, child in node.children.items():
        vals[action] = child.default_value()

    return transform.normalize(vals)



