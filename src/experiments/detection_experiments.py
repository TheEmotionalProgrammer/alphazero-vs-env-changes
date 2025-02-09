
"""

#############################
##  DETECTION EXPERIMENTS  ##
#############################

This script can be used to test the detection accuracy of AZDetection in different settings.

"""



import sys

sys.path.append("src/")

import json
import os
from tqdm import tqdm
import numpy as np
import multiprocessing
import gymnasium as gym
import wandb
import pandas as pd
import matplotlib.pyplot as plt

from log_code.metrics import calc_metrics
from experiments.eval_agent import eval_agent
from experiments.evaluate_from_config import agent_from_config
from environments.observation_embeddings import ObservationEmbedding, embedding_dict
from az.azmcts import AlphaZeroMCTS
from azdetection.change_detector import AlphaZeroDetector
from azdetection.octopus import Octopus
from az.model import (
    AlphaZeroModel,
    models_dict
)
from policies.tree_policies import tree_eval_dict
from policies.selection_distributions import selection_dict_fn
from policies.value_transforms import value_transform_dict
from policies.policies import PolicyDistribution

import torch as th

import copy
from environments.register import register_all

import argparse
from parameters import base_parameters, env_challenges, fz_env_descriptions

def run_single_detection(
        solver: AlphaZeroDetector,
        env: gym.Env,
        tree_evaluation_policy: PolicyDistribution,
        observation_embedding: ObservationEmbedding,
        planning_budget = 1000,
        max_steps = 1000,
        seed=None,
        temperature=None,
        original_env: gym.Env | None = None,
        unroll_budget=5,
    ):

    """
    Run detection for a single configuration.
    We stop as soon as we detect the obstacle and return:
    - the number of steps from the root to the obstacle (None if no detection, 'None' in the CSV file)
    - the number of steps taken before the detection is triggered (inf if no detection, 'inf' in the CSV file)
    """

    assert isinstance(env.action_space, gym.spaces.Discrete) # For now, only supports discrete action spaces
    n = int(env.action_space.n)

    if seed is not None:
        th.manual_seed(seed)
        np.random.seed(seed)

    observation, _ = env.reset(seed=seed)

    old_obs = observation

    if original_env is not None:
        _, _ = original_env.reset(seed=seed) 
        original_env.unwrapped.s = env.unwrapped.s # Set the initial state of the original environment to match the current environment

    tree = solver.search(env,planning_budget, observation, 0.0, original_env = original_env, n= unroll_budget, env_action=None)

    step = 0

    if solver.problem_idx is not None:
        distance_from_obst = solver.problem_idx 
        distance_from_init = solver.root_idx
        return distance_from_obst, distance_from_init # For accuracy and sensitivity computations

    while step < max_steps:

        tree.reset_var_val() # The value and variance (mvc) estimates of the whole subtree are reset.

        action = th.argmax(tree.prior_policy).item()
    
        new_obs, reward, terminated, truncated, _ = env.step(action)

        new_pos_row = new_obs // observation_embedding.ncols
        new_pos_col = new_obs % observation_embedding.ncols

        print(f"Env: obs = ({new_pos_row}, {new_pos_col}), reward = {reward}, terminated = {terminated}, truncated = {truncated}")

        if original_env is not None:
                if new_obs != old_obs:
                    _, _, _, _, _ = original_env.step(action)

        old_obs = new_obs

        next_terminal = terminated

        if next_terminal or truncated:
            break

        tree = solver.search(env, planning_budget, new_obs, reward, original_env=original_env, n=unroll_budget, env_action=action)

        if solver.problem_idx is not None:
            distance_from_obst = solver.problem_idx - solver.root_idx
            distance_from_init = solver.root_idx
            return distance_from_obst, distance_from_init

        step += 1
    
    # No obstacle detected
    return None, "inf"


def run_all(run_configs, num_seeds):

    """
    Run detection for all configurations and save results in a dict and saved as a JSON.
    The structure of the dict will be as follows:

    "env_config_name_1": {
        "train_seed_1": {
            "detection_steps": detection_steps,
            "steps_before_detection": steps_before_detection
        },
        "train_seed_2": {
            "detection_steps": detection_steps,
            "steps_before_detection": steps_before_detection
        },
        ...
    "env_config_name_2": {
        ...
    }

    """

    results = {}

    for run_config in run_configs:
        run_config = run_configs[run_config]
        config_copy = dict(run_config)

        test_env = gym.make(**config_copy["test_env"])

        name_config = config_copy["map_name"]

        results[name_config] = {}

        for model_seed in range(num_seeds):

            model_file = f"hyper/AZTrain_env=CustomFrozenLakeNoHoles16x16-v1_evalpol=visit_iterations=60_budget=128_df=0.95_lr=0.003_nstepslr=2_seed={model_seed}/checkpoint.pth"

            config_copy["model_file"] = model_file

            (
            agent,
            train_env, 
            tree_evaluation_policy, 
            observation_embedding, 
            planning_budget
            ) = agent_from_config(config_copy)

            detection_steps, steps_before_detection = run_single_detection(
                solver=agent,
                env=test_env,
                tree_evaluation_policy=tree_evaluation_policy,
                observation_embedding=observation_embedding,
                planning_budget=planning_budget,
                max_steps=1000,
                seed=0,
                temperature=config_copy["eval_temp"],
                original_env=train_env,
                unroll_budget=config_copy["unroll_budget"],
            )

            results[name_config][model_seed] = {
                "distance_init_obst": detection_steps,
                "distance_from_init": steps_before_detection
            }

    return results


if __name__ == "__main__":

    register_all()

    parser = argparse.ArgumentParser(description="AlphaZero Evaluation Configuration")

    # AZDetection detection parameters
    parser.add_argument("--threshold", type=float, default=0.05, help="Detection threshold")
    parser.add_argument("--unroll_budget", type=int, default=4, help="Unroll budget")

    parser.add_argument("--train_seeds", type=int, default=10, help="The number of random seeds to use for training.")
    parser.add_argument("--eval_seeds", type=int, default=1, help="The number of random seeds to use for evaluation.")

    parser.add_argument("--test_env_is_slippery", type=bool, default=False, help="Slippery environment")
    parser.add_argument("--test_env_hole_reward", type=int, default=0, help="Hole reward")
    parser.add_argument("--test_env_terminate_on_hole", type=bool, default=False, help="Terminate on hole")
    parser.add_argument("--deviation_type", type=str, default="bump", help="Deviation type")

    parser.add_argument("--value_estimate", type=str, default="nn", help="Value estimate")
    parser.add_argument("--predictor", type=str, default="current_value", help="Predictor")
    parser.add_argument("--update_estimator", type=bool, default=True, help="Update estimator")

    # Parse arguments
    args = parser.parse_args()

    # Add fixed parameters (do not modify)

    args.wandb_logs = False
    args.workers = 1
    args.runs = 1
    args.tree_evaluation_policy = "mvc"
    args.selection_policy = "PolicyUCT"
    args.planning_budget = args.unroll_budget + 1
    args.puct_c = 1.0
    args.agent_type = "azdetection"
    args.eval_temp = 0.0
    args.dir_epsilon = 0.0
    args.dir_alpha = None
    args.planning_style = "connected"
    args.value_search = False
    args.test_env_id = "CustomFrozenLakeNoHoles16x16-v1"
    args.observation_embedding = "coordinate"
    args.render = False
    args.hpc = False
    args.map_size = 16
    args.visualize_trees = False
    args.var_penalty = 1.0

    challenge = env_challenges["CustomFrozenLakeNoHoles16x16-v1"]  # Training environment

    # Construct the config
    config_modifications = {
        "wandb_logs": args.wandb_logs,
        "workers": args.workers,
        "runs": args.runs,
        "tree_evaluation_policy": args.tree_evaluation_policy,
        "selection_policy": args.selection_policy,
        "planning_budget": args.planning_budget,
        "puct_c": args.puct_c,
        "agent_type": args.agent_type,
        "eval_temp": args.eval_temp,
        "dir_epsilon": args.dir_epsilon,
        "dir_alpha": args.dir_alpha,
        "threshold": args.threshold,
        "unroll_budget": args.unroll_budget,
        "planning_style": args.planning_style,
        "value_search": args.value_search,
        "observation_embedding": args.observation_embedding,
        "render": args.render,
        "hpc": args.hpc,
        "visualize_trees": args.visualize_trees,
        "map_size": args.map_size,
        "var_penalty": args.var_penalty,

        "value_estimate": args.value_estimate,
        "predictor": args.predictor,
        "update_estimator": args.update_estimator
        
    }

    run_config = {**base_parameters, **challenge, **config_modifications}

    test_env_configs = ["16x16_NO_OBSTACLES", "16x16_D2", "16x16_D5", "16x16_D8", "16x16_D11", "16x16_D14"]

    test_env_config_dict = {
        key: {
            "id": args.test_env_id,
            "desc": fz_env_descriptions[key],
            "is_slippery": args.test_env_is_slippery,
            "hole_reward": args.test_env_hole_reward,
            "terminate_on_hole": args.test_env_terminate_on_hole,
            "deviation_type": args.deviation_type
        }
        for key in test_env_configs
    }

    run_configs = {}

    for test_env_config in test_env_configs:
        
        run_config["test_env"] = {
            "id": test_env_config_dict[test_env_config]["id"],
            "desc": test_env_config_dict[test_env_config]["desc"],
            "is_slippery": test_env_config_dict[test_env_config]["is_slippery"],
            "hole_reward": test_env_config_dict[test_env_config]["hole_reward"],
            "terminate_on_hole": test_env_config_dict[test_env_config]["terminate_on_hole"],
            "deviation_type": test_env_config_dict[test_env_config]["deviation_type"]
        }
        run_config["map_name"] = test_env_config

        run_configs[test_env_config] = copy.deepcopy(run_config)

    detection_params_combinations = [
        {
            "value_estimate": "nn",
            "predictor": "current_value",
            "update_estimator": True
        },
        {
            "value_estimate": "nn",
            "predictor": "current_value",
            "update_estimator": False
        },
        {
            "value_estimate": "nn",
            "predictor": "original_env",
            "update_estimator": False
        }
    ]

    run_configs_combinations = []

    for detection_params in detection_params_combinations:
        strkeys = "__".join([f"{key}={detection_params[key]}" for key in detection_params])
        for run_config in run_configs:
            for key in detection_params:
                run_configs[run_config][key] = detection_params[key]

        results = run_all(run_configs, args.train_seeds)

        # If the directory does not exist, create it
        if not os.path.exists("detection_results"):
            os.makedirs("detection_results")

        # Save results to a JSON file
        with open(f"detection_results/{strkeys}.json", "w") as f:
            json.dump(results, f)



    




    