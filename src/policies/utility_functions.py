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
    Computes the Q estimates yielded by the evaluation policy that we are currently using.
    For example, the default visit count evaluator induces the arithmetic mean Q estimate.
    """

    # If the node is terminal, its value is just the reward.
    if node.terminal:
        val = th.tensor(node.reward, dtype=th.float32)
        node.policy_value = val
        return val

    # If the value has already been computed, return it.
    if node.policy_value is not None:
        return node.policy_value
    
    if isinstance(policy, th.distributions.Categorical):
        pi = policy
    else:
        pi = policy.softmaxed_distribution(node, include_self=True)

    probabilities: th.Tensor = pi.probs
    #print("probabilities", probabilities)
    assert probabilities.shape[-1] == int(node.action_space.n) + 1
    own_propability = probabilities[-1]  # type: ignore
    child_propabilities = probabilities[:-1]  # type: ignore
    child_values = th.zeros_like(child_propabilities, dtype=th.float32) # For all the unexpanded children, the value is 0
    #print("child_values_before", child_values)
    
    # We recursively compute the value estimates of the children 
    for action, child in node.children.items():
        child_values[action] = policy_value(child, policy, discount_factor) 
   
    #print("child_values", child_values)
    #print(node.value_evaluation)
    val = node.reward + discount_factor * (
        own_propability * node.value_evaluation
        + (child_propabilities * child_values).sum()
    )
   
    node.policy_value = val

    return val


def reward_variance(node: Node):

    """
    The variance of the reward of the node, under the assumption that it is deterministic.
    """

    return 0.0

def value_evaluation_variance(node: Node):

    if node.terminal:
        return 1 / float(node.visits)

    if node.problematic:

        return node.var_penalty

    return 1
    
def independent_policy_value_variance(
    node: Node,
    policy: PolicyDistribution | th.distributions.Categorical,
    discount_factor: float,
):
    if node.variance is not None:
        return node.variance
    # return the variance of the q value the node with the given policy
    if isinstance(policy, th.distributions.Categorical):
        pi = policy
    else:
        pi = policy.softmaxed_distribution(node, include_self=True)

    probabilities_squared = pi.probs**2  # type: ignore
    own_propability_squared = probabilities_squared[-1]
    child_propabilities_squared = probabilities_squared[:-1]
    child_variances = th.zeros_like(child_propabilities_squared, dtype=th.float32)

    for action, child in node.children.items():
        child_variances[action] = independent_policy_value_variance(
            child, policy, discount_factor
        )

    var = reward_variance(node) + discount_factor**2 * (
        own_propability_squared * value_evaluation_variance(node)
        + (child_propabilities_squared * child_variances).sum()
    )

    node.variance = var
    return var


def get_children_policy_values(
    parent: Node,
    policy: PolicyDistribution,
    discount_factor: float,
    transform: ValueTransform = IdentityValueTransform,
) -> th.Tensor:
    # have a look at this, infs mess things up
    vals = th.ones(int(parent.action_space.n), dtype=th.float32) * -th.inf
    for action, child in parent.children.items():
        vals[action] = policy_value(child, policy, discount_factor)
    vals = transform.normalize(vals)

    return vals

def get_children_q_max_values(
    parent: Node,
    discount_factor: float,
    transform: ValueTransform = IdentityValueTransform,   
):
    vals = th.ones(int(parent.action_space.n), dtype=th.float32) * -th.inf
    for action, child in parent.children.items():
        vals[action] = child.reward + discount_factor * q_max_value(child, discount_factor)
    vals = transform.normalize(vals)

    return vals

def q_max_value(node: Node, discount_factor: float):

    if node.terminal:
        return node.reward
    
    if node.children == {}:
        return node.value_evaluation
    
    children_values = [q_max_value(child, discount_factor) for child in node.children.values()]

    values = children_values #+ [self_value]

    return node.reward + discount_factor * max(values)
        
def get_children_inverse_variances(
    parent: Node, policy: PolicyDistribution, discount_factor: float
) -> th.Tensor:
    inverse_variances = th.zeros(int(parent.action_space.n), dtype=th.float32)
    for action, child in parent.children.items():
        inverse_variances[action] = 1.0 / independent_policy_value_variance(
            child, policy, discount_factor
        )

    return inverse_variances

def get_children_variances(
    parent: Node, policy: PolicyDistribution, discount_factor: float
) -> th.Tensor:
    variances = th.ones(int(parent.action_space.n), dtype=th.float32)
    for action, child in parent.children.items():
        variances[action] = independent_policy_value_variance(
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
    This is more efficent than calling get_children_policy_values and get_children_variances separately
    """
    vals = th.ones(int(parent.action_space.n) + include_self, dtype=th.float32) * -th.inf
    inv_vars = th.zeros_like(vals + include_self, dtype=th.float32)
    for action, child in parent.children.items():
        pi = policy#.softmaxed_distribution(child, include_self=True)
        #print("pi", pi.probs)
        vals[action] = policy_value(child, pi, discount_factor)
        inv_vars[action] = 1 / independent_policy_value_variance(
            child, pi, discount_factor
        )
    if include_self:
        vals[-1] = parent.value_evaluation
        inv_vars[-1] = 1 / value_evaluation_variance(parent)

    normalized_vals = transform.normalize(vals)
    #print(inv_vars)
    return normalized_vals, inv_vars

def expanded_mask(node: Node) -> th.Tensor:
    mask = th.zeros(int(node.action_space.n), dtype=th.float32)
    mask[node.children] = 1.0
    return mask

def get_children_visits(node: Node) -> th.Tensor:
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

def puct_multiplier(c: float, node: Node):

    """
    lambda_N from the mcts as policy optimization paper.
    """

    return c * (node.visits**0.5) / (node.visits + int(node.action_space.n))


def Q_theta_tensor(node: Node, discount: float, transform: ValueTransform = IdentityValueTransform) -> th.Tensor:
    """
    Returns a vector with the Q_theta values. Zero for unvisited nodes, for visited nodes it is return + the discoutned value evaluation of the node
    """
    vals = th.zeros(int(node.action_space.n), dtype=th.float32)
    for action, child in node.children.items():
        val = .0 if child.is_terminal() else child.value_evaluation
        vals[action] = child.reward + discount * val
    return transform.normalize(vals)
