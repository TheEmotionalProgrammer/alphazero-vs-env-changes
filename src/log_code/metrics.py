import torch as th
import numpy as np

def calc_metrics(tensor_res: th.Tensor, discount_factor: float, n: int, epsilon = 1e-8):
    episode_returns = th.sum(tensor_res["rewards"]* tensor_res["mask"], dim=-1)
    discounts = discount_factor ** th.cumsum(tensor_res["mask"], dim=-1)
    discounted_returns = th.sum(tensor_res["rewards"] * tensor_res["mask"] * discounts, dim=-1)
    time_steps = th.sum(tensor_res["mask"], dim=-1, dtype=th.float32)
    entropies = -th.sum(
            tensor_res["policy_distributions"]
            * th.log(tensor_res["policy_distributions"] + epsilon),
            dim=-1,
        ) * tensor_res["mask"] / np.log(n)
    return episode_returns, discounted_returns, time_steps, entropies

