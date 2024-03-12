import torch as th

from core.node import Node
from policies.policies import PolicyDistribution
from policies.utility_functions import basic_value_normalizer, independent_policy_value_variance, policy_value, puct_multiplier, value_evaluation_variance




class VistationPolicy(PolicyDistribution):
    # the default tree evaluator selects the action with the most visits
    def _distribution(self, node: Node, include_self = False) -> th.distributions.Categorical:
        visits = th.zeros(int(node.action_space.n) + include_self)
        for action, child in node.children.items():
            visits[action] = child.visits

        if include_self:
            visits[-1] = 1

        return th.distributions.Categorical(visits)


class InverseVarianceTreeEvaluator(PolicyDistribution):
    """
    Selects the action with the highest inverse variance of the q value.
    Should return the same as the default tree evaluator
    """
    def __init__(self, discount_factor = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount_factor = discount_factor

    def _distribution(self, node: Node, include_self=False) -> th.distributions.Categorical:
        inverse_variances = th.zeros(int(node.action_space.n) + include_self, dtype=th.float32)

        for action, child in node.children.items():
            inverse_variances[action] = 1.0 / independent_policy_value_variance(child, self, self.discount_factor)


        if include_self:
            # if we have no children, the policy should be 1 for the self action
            if len(node.children) == 0:
                inverse_variances[-1] = 1.0
            else:
                # set it to 1/visits
                inverse_variances[-1] = inverse_variances.sum() / (node.visits - 1)

        return th.distributions.Categorical(
            inverse_variances
        )


# minimal-variance constraint policy
class MinimalVarianceConstraintPolicy(PolicyDistribution):
    """
    Selects the action with the highest inverse variance of the q value.
    Should return the same as the default tree evaluator
    """
    def __init__(self, beta: float, discount_factor = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.discount_factor = discount_factor

    def get_beta(self, node: Node):
        return self.beta

    def _distribution(self, node: Node, include_self=False) -> th.distributions.Categorical:
        if len(node.children) == 0:
            d = th.zeros(int(node.action_space.n) + include_self, dtype=th.float32)
            d[-1] = 1.0
            return th.distributions.Categorical(d)

        beta = self.get_beta(node)

        # have a look at this, infs mess things up
        vals = th.ones(int(node.action_space.n), dtype=th.float32) * - th.inf
        inv_vars = th.zeros_like(vals, dtype=th.float32)
        for action, child in node.children.items():
            pi = self.softmaxed_distribution(child, include_self=True)
            vals[action] = policy_value(child, pi, self.discount_factor)
            inv_vars[action] = 1/independent_policy_value_variance(child, pi, self.discount_factor)

        normalized_vals = basic_value_normalizer(vals)
        policy = th.zeros(int(node.action_space.n) + include_self, dtype=th.float32)

        for action, child in node.children.items():
            policy[action] = th.exp(beta * (normalized_vals[action] - normalized_vals.max())) * inv_vars[action]

        # if include_self:
        #     # TODO: this probably has to be updated
        #     vals[-1] = th.tensor(node.value_evaluation)
        #     inv_vars[-1] = 1.0 / (self.discount_factor ** 2 * value_evaluation_variance(node))

        # risk for numerical instability if vals are large/small
        # Solution: subtract the mean
        # This should be equivalent to muliplying by a constant which we can ignore
        # policy = th.exp(self.beta * (vals - vals.max())) * inv_vars


        # make a numerical check. If the policy is all 0, then we should return a uniform distribution
        # if policy.sum() == 0:
        #     policy[:-1] = th.ones_like(policy[:-1])


        if include_self:
            # if we have no children, the policy should be 1 for the self action
            if len(node.children) == 0:
                policy[-1] = 1.0
            else:
                # set it to 1/visits
                policy[-1] = policy.sum() / (node.visits - 1)

        # # if we have some infinities, we should return a uniform distribution over the infinities
        # if th.isinf(policy).any():
        #     return th.distributions.Categorical(
        #         th.isinf(policy)
        #     )

        return th.distributions.Categorical(
            policy
        )

class MVCP_Dynamic_Beta(MinimalVarianceConstraintPolicy):
    """
    From the mcts as policy optimization paper
    """
    def __init__(self, c: float, discount_factor=1, *args, **kwargs):
        super(MinimalVarianceConstraintPolicy, self).__init__(*args, **kwargs)
        self.c = c
        self.discount_factor = discount_factor

    def get_beta(self, node: Node):
        return 1 / puct_multiplier(self.c, node)

class ReversedRegPolicy(PolicyDistribution):
    """
    From the mcts as policy optimization paper
    """
    def __init__(self, c: float, discount_factor=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c
        self.discount_factor = discount_factor


    def _distribution(self, node: Node, include_self=False) -> th.distributions.Categorical:
        if len(node.children) == 0:
            d = th.zeros(int(node.action_space.n) + include_self, dtype=th.float32)
            d[-1] = 1.0
            return th.distributions.Categorical(d)

        # have a look at this, infs mess things up
        vals = th.ones(int(node.action_space.n) + include_self, dtype=th.float32) * - th.inf
        for action, child in node.children.items():
            vals[action] = policy_value(child, self, self.discount_factor)
        vals = basic_value_normalizer(vals)
        policy = th.zeros_like(vals, dtype=th.float32)
        mult = puct_multiplier(self.c, node)

        for action, child in node.children.items():
            policy[action] = node.prior_policy[action] * th.exp((vals[action] - vals.max()) / mult)

        if include_self:
            # if we have no children, the policy should be 1 for the self action
            if len(node.children) == 0:
                policy[-1] = 1.0
            else:
                # set it to 1/visits
                policy[-1] = policy.sum() / (node.visits - 1)


        return th.distributions.Categorical(
            policy
        )


class MVTOPolicy(PolicyDistribution):
    """
    Solution to argmax mu + lambda * var
    """

    def __init__(self, lamb: float, discount_factor=1, *args, **kwargs):
        """
        Note that lambda > max_i sum_j (var_j^-1 ( x_j - x_i))/ 2
        """
        super().__init__(*args, **kwargs)
        self.lamb = lamb
        self.discount_factor = discount_factor


    def _distribution(self, node: Node, include_self=False) -> th.distributions.Categorical:
        if len(node.children) == 0:
            d = th.zeros(int(node.action_space.n) + include_self, dtype=th.float32)
            d[-1] = 1.0
            return th.distributions.Categorical(d)


        # have a look at this, infs mess things up
        vals = th.zeros(int(node.action_space.n) + include_self, dtype=th.float32)
        inv_vars = th.zeros_like(vals, dtype=th.float32)
        for action, child in node.children.items():
            pi = self.softmaxed_distribution(child, include_self=True)
            vals[action] = policy_value(child, pi, self.discount_factor)
            inv_vars[action] = 1/independent_policy_value_variance(child, pi, self.discount_factor)

        vals = basic_value_normalizer(vals)
        inv_var_policy = inv_vars / inv_vars.sum()

        policy = th.zeros_like(vals, dtype=th.float32)

        piv_sum = (inv_var_policy * vals).sum()
        a = .5 / self.lamb

        for action, child in node.children.items():
            policy[action] = inv_var_policy[action] + a * inv_vars[action] * (vals[action] - piv_sum)

        if policy.min() < 0:
            # seems like lambda is too small, follow the greedy policy instead
            # alternativly we could just set the negative values to 0
            print("lambda too small, using greedy policy")
            g =  GreedyPolicy(self.discount_factor).softmaxed_distribution(node, include_self)
            return g


        if include_self:
            # if we have no children, the policy should be 1 for the self action
            if len(node.children) == 0:
                policy[-1] = 1.0
            else:
                # set it to 1/visits
                policy[-1] = policy.sum() / (node.visits - 1)


        return th.distributions.Categorical(
            policy
        )


class GreedyPolicy(PolicyDistribution):
    def __init__(self, discount_factor = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount_factor = discount_factor

    """
    Determinstic policy that selects the action with the highest value
    """
    def _distribution(self, node: Node, include_self=False) -> th.distributions.Categorical:
        if len(node.children) == 0:
            d = th.zeros(int(node.action_space.n) + include_self, dtype=th.float32)
            d[-1] = 1.0
            return th.distributions.Categorical(d)

        vals = th.ones(int(node.action_space.n) + include_self, dtype=th.float32) * - th.inf

        for action, child in node.children.items():
            vals[action] = policy_value(child, self, self.discount_factor)


        max_val = vals.max()
        max_mask = (vals == max_val)
        policy = th.zeros_like(vals)
        policy[max_mask] = 1.0


        if include_self:
            # if we have no children, the policy should be 1 for the self action
            if len(node.children) == 0:
                policy[-1] = 1.0
            else:
                # set it to 1/visits
                policy[-1] = policy.sum() / (node.visits - 1)


        return th.distributions.Categorical(
            policy
        )

tree_dict = {
    "visit": VistationPolicy,
    "inverse_variance": InverseVarianceTreeEvaluator,
    "mvc": MinimalVarianceConstraintPolicy,
    'mvc_dynbeta': MVCP_Dynamic_Beta,
    'reversedregpolicy': ReversedRegPolicy,
    "mvto": MVTOPolicy,
}

tree_eval_dict = lambda param, discount, c=1.0, temperature=None: {
    "visit": VistationPolicy(temperature),
    "inverse_variance": InverseVarianceTreeEvaluator(discount_factor=discount, temperature=temperature),
    "mvc": MinimalVarianceConstraintPolicy(discount_factor=discount, beta=param, temperature=temperature),
    'mvc_dynbeta': MVCP_Dynamic_Beta(c=c, discount_factor=discount, temperature=temperature),
    'reversedregpolicy': ReversedRegPolicy(c=c, discount_factor=discount, temperature=temperature),
    "mvto": MVTOPolicy(lamb=param, discount_factor=discount, temperature=temperature),
}
