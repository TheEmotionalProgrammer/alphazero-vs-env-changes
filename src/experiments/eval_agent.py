from typing import List
from core.mcts import MCTS
import gymnasium as gym

from core.runner import collect_trajectories
from environments.observation_embeddings import ObservationEmbedding
from log_code.metrics import calc_metrics
from policies.policies import PolicyDistribution
import torch as t


def eval_agent(agent: MCTS, env: gym.Env, tree_evaluation_policy: PolicyDistribution, observation_embedding: ObservationEmbedding, planning_budget: int, max_episode_length: int, seeds: List[int | None], temperature: float | None, workers=1, azdetection=False, original_env: gym.Env | None = None, unroll_budget=5):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    tasks = [(agent, env, tree_evaluation_policy, observation_embedding, planning_budget, max_episode_length, seed, temperature, azdetection, original_env, unroll_budget) for seed in seeds]
    results = collect_trajectories(
        tasks,
        workers=workers,
    )
    return results
