import sys

sys.path.append("src/")

import os
import numpy as np
import torch as th
import multiprocessing
import gymnasium as gym
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from log_code.metrics import calc_metrics
from experiments.evaluation.eval_agent import eval_agent
from core.mcts import RandomRolloutMCTS
from environments.observation_embeddings import ObservationEmbedding, embedding_dict
from az.azmcts import AlphaZeroMCTS, AlphaZeroNoLoops

from policies.utility_functions import policy_value

from azdetection.minitrees import MiniTrees
from azdetection.megatree import MegaTree
from azdetection.pddp import PDDP

from experiments.extra.create_gif import create_gif

from az.model import (
    AlphaZeroModel,
    models_dict
)

from policies.tree_policies import tree_eval_dict
from policies.selection_distributions import selection_dict_fn

from policies.value_transforms import value_transform_dict

from environments.register import register_all

import argparse
from experiments.parameters import base_parameters, env_challenges, fz_env_descriptions, ll_env_descriptions

from log_code.tree_visualizer import visualize_trees
from log_code.plot_state_densities import plot_density, calculate_density, calculate_nn_value_means, calculate_policy_value_means, calculate_nn_value_sums, calculate_repetitions, calculate_policy_value_sums, plot_density_values_only
from experiments.evaluation.evaluate_from_config import agent_from_config

def invert_fz_coordinate(coords, map_size):
    return coords[0] * map_size + coords[1] if coords is not None else None

def create_density_visualizations(run_config, root_state, eval_seed=8):

    hparams = run_config

    agent, _, tree_eval_policy, _, planning_budget = (
        agent_from_config(hparams)
    )

    test_env = gym.make(**hparams["test_env"])

    if eval_seed is not None:
        th.manual_seed(eval_seed)
        np.random.seed(eval_seed)

    test_env.reset(seed=eval_seed)

    test_env.unwrapped.s = invert_fz_coordinate(root_state, hparams["map_size"])
    observation = test_env.unwrapped.s

    tree = agent.search(test_env, planning_budget, observation,  0)

    test_desc = hparams["test_env"]["desc"]
    obst_coords = []
    for i in range(len(test_desc)):
        for j in range(len(test_desc[0])):
            if test_desc[i][j] == "H":
                obst_coords.append((i, j))

    policy_value(tree, tree_eval_policy, hparams["discount_factor"])

    policy_values = calculate_policy_value_means(tree, map_size, map_size)

    values_cmap = sns.diverging_palette(240, 10, as_cmap=True, center="light")

    ax = plot_density(policy_values, tree.observation, obst_coords, map_size, map_size, cmap="gray")

    ax.set_title(f"Mean Policy Values")

    # If no path to folder, create it
    # if not os.path.exists("policy_values"):
    #     os.makedirs("policy_values")

    # plt.savefig(f"policy_values/.png")
    plt.show()
    plt.close()

    # nn_values = calculate_nn_value_means(tree, map_size, map_size)
    # values_cmap = sns.diverging_palette(120, 10, as_cmap=True, center="light")


    # ax = plot_density(nn_values, tree.observation, obst_coords, map_size, map_size, cmap=values_cmap)
    # ax.set_title(f"Mean NN Values")
    # # If no path to folder, create it
    # # if not os.path.exists("nn_values"):
    # #     os.makedirs("nn_values")
    # plt.show()
    # plt.close()

def create_average_density_visualizations(run_config, root_state, train_seeds, eval_seeds):
    """
    Create density visualizations averaged across multiple training and evaluation seeds.

    Args:
        run_config (dict): Configuration for the run.
        root_state (tuple): Root state coordinates.
        train_seeds (list): List of training seeds.
        eval_seeds (list): List of evaluation seeds.
    """
    hparams = run_config
    map_size = hparams["map_size"]

    # Initialize accumulators for averaging
    accumulated_policy_values = None
    accumulated_counts = None
    accumulated_visits = None

    for train_seed in train_seeds:
        # Set model_file based on map_size and train_config
        if hparams["map_size"] == 8 and hparams["train_config"] == "NO_HOLES":
            hparams["model_file"] = f"hyper/AZTrain_env=CustomFrozenLakeNoHoles8x8-v1_evalpol=visit_iterations=50_budget=64_df=0.95_lr=0.001_nstepslr=2_seed={train_seed}/checkpoint.pth"
        elif hparams["map_size"] == 16 and hparams["train_config"] == "NO_HOLES":
            hparams["model_file"] = f"hyper/AZTrain_env=CustomFrozenLakeNoHoles16x16-v1_evalpol=visit_iterations=60_budget=128_df=0.95_lr=0.003_nstepslr=2_seed={train_seed}/checkpoint.pth"
        elif hparams["map_size"] == 8 and hparams["train_config"] == "MAZE_RL":
            if hparams["bad_training"]:
                hparams["model_file"] = f"hyper/AZTrain_env=8x8_MAZE_RL_evalpol=visit_iterations=20_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={train_seed}/checkpoint.pth"
            else:
                hparams["model_file"] = f"hyper/AZTrain_env=8x8_MAZE_RL_evalpol=visit_iterations=150_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={train_seed}/checkpoint.pth"
        elif hparams["map_size"] == 8 and hparams["train_config"] == "MAZE_LR":
            if hparams["bad_training"]:
                hparams["model_file"] = f"hyper/AZTrain_env=8x8_MAZE_LR_evalpol=visit_iterations=10_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={train_seed}/checkpoint.pth"
            else:
                hparams["model_file"] = f"hyper/AZTrain_env=8x8_MAZE_LR_evalpol=visit_iterations=100_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={train_seed}/checkpoint.pth"
        elif hparams["map_size"] == 16 and hparams["train_config"] == "MAZE_LR":
            hparams["model_file"] = f"hyper/AZTrain_env=16x16_MAZE_LR_evalpol=visit_iterations=150_budget=64_df=0.95_lr=0.003_nstepslr=2_c=0.2_seed={train_seed}/checkpoint.pth"

        for eval_seed in eval_seeds:
            agent, _, tree_eval_policy, _, planning_budget = (
                agent_from_config(hparams)
            )

            test_env = gym.make(**hparams["test_env"])

            if eval_seed is not None:
                th.manual_seed(eval_seed)
                np.random.seed(eval_seed)

            test_env.reset(seed=eval_seed)

            test_env.unwrapped.s = invert_fz_coordinate(root_state, hparams["map_size"])
            observation = test_env.unwrapped.s

            tree = agent.search(test_env, planning_budget, observation, 0)

            test_desc = hparams["test_env"]["desc"]
            obst_coords = []
            for i in range(len(test_desc)):
                for j in range(len(test_desc[0])):
                    if test_desc[i][j] == "H":
                        obst_coords.append((i, j))

            policy_value(tree, tree_eval_policy, hparams["discount_factor"])

            policy_values = calculate_policy_value_sums(tree, map_size, map_size)
            counts = calculate_repetitions(tree, map_size, map_size)
            visits  = calculate_density(tree, map_size, map_size)

            # Accumulate policy values
            if accumulated_policy_values is None:
                accumulated_policy_values = policy_values
            else:
                accumulated_policy_values += policy_values

            # # Accumulate counts
            if accumulated_counts is None:
                accumulated_counts = counts
            else:
                accumulated_counts += counts

            if accumulated_visits is None:
                accumulated_visits = visits
            else:
                accumulated_visits += visits

    # Safely divide accumulated_policy_values by counts, avoiding division by zero
    # if accumulated_policy_values is not None:
    #     averaged_policy_values = np.divide(
    #         accumulated_policy_values,
    #         accumulated_counts,
    #         out=np.zeros_like(accumulated_policy_values),  # Default value for zero counts
    #         where=accumulated_counts != 0  # Perform division only where counts are non-zero
    #     )
    #     # Set unexplored states to np.nan so they are not visualized
    #     explored_mask = accumulated_counts != 0
    #     averaged_policy_values[~explored_mask] = np.nan

    #     #policy_values_cmap = sns.diverging_palette(10, 240, as_cmap=True, center="light")
    #     #ax = plot_density(averaged_policy_values, tree.observation, obst_coords, map_size, map_size, cmap=policy_values_cmap)
    #     ax = plot_density_values_only(averaged_policy_values, tree.observation, obst_coords, map_size, map_size)
    #     #ax.set_title(f"Averaged Mean Policy Values Across Training and Evaluation Seeds")
    #     plt.show()
    #     plt.close()

    # Plot the accumulated visits

    visits_cmap = sns.diverging_palette(10, 120, as_cmap=True, center="light")
    ax = plot_density(accumulated_visits/100, tree.observation, obst_coords, map_size, map_size, cmap=visits_cmap)
    #ax.set_title(f"Averaged Visits Across Training and Evaluation Seeds")
    plt.show()
    plt.close()

def compare_average_density_visualizations(run_config1, run_config2, root_state, train_seeds, eval_seeds):
    """
    Compute and plot the difference between average density visualizations for two run_configs,
    masking out states not explored in both runs.
    """
    def get_averaged_policy_values(run_config):
        hparams = run_config
        map_size = hparams["map_size"]
        accumulated_policy_values = None
        accumulated_counts = None

        for train_seed in train_seeds:
            # Set model_file as in create_average_density_visualizations
            if hparams["map_size"] == 8 and hparams["train_config"] == "NO_HOLES":
                hparams["model_file"] = f"hyper/AZTrain_env=CustomFrozenLakeNoHoles8x8-v1_evalpol=visit_iterations=50_budget=64_df=0.95_lr=0.001_nstepslr=2_seed={train_seed}/checkpoint.pth"
            elif hparams["map_size"] == 16 and hparams["train_config"] == "NO_HOLES":
                hparams["model_file"] = f"hyper/AZTrain_env=CustomFrozenLakeNoHoles16x16-v1_evalpol=visit_iterations=60_budget=128_df=0.95_lr=0.003_nstepslr=2_seed={train_seed}/checkpoint.pth"
            elif hparams["map_size"] == 8 and hparams["train_config"] == "MAZE_RL":
                if hparams["bad_training"]:
                    hparams["model_file"] = f"hyper/AZTrain_env=8x8_MAZE_RL_evalpol=visit_iterations=20_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={train_seed}/checkpoint.pth"
                else:
                    hparams["model_file"] = f"hyper/AZTrain_env=8x8_MAZE_RL_evalpol=visit_iterations=150_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={train_seed}/checkpoint.pth"
            elif hparams["map_size"] == 8 and hparams["train_config"] == "MAZE_LR":
                if hparams["bad_training"]:
                    hparams["model_file"] = f"hyper/AZTrain_env=8x8_MAZE_LR_evalpol=visit_iterations=10_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={train_seed}/checkpoint.pth"
                else:
                    hparams["model_file"] = f"hyper/AZTrain_env=8x8_MAZE_LR_evalpol=visit_iterations=100_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={train_seed}/checkpoint.pth"
            elif hparams["map_size"] == 16 and hparams["train_config"] == "MAZE_LR":
                hparams["model_file"] = f"hyper/AZTrain_env=16x16_MAZE_LR_evalpol=visit_iterations=150_budget=64_df=0.95_lr=0.003_nstepslr=2_c=0.2_seed={train_seed}/checkpoint.pth"

            for eval_seed in eval_seeds:
                agent, _, tree_eval_policy, _, planning_budget = (
                    agent_from_config(hparams)
                )
                test_env = gym.make(**hparams["test_env"])
                if eval_seed is not None:
                    th.manual_seed(eval_seed)
                    np.random.seed(eval_seed)
                test_env.reset(seed=eval_seed)
                test_env.unwrapped.s = invert_fz_coordinate(root_state, hparams["map_size"])
                observation = test_env.unwrapped.s
                tree = agent.search(test_env, planning_budget, observation, 0)
                policy_value(tree, tree_eval_policy, hparams["discount_factor"])
                policy_values = calculate_policy_value_sums(tree, map_size, map_size)
                counts = calculate_repetitions(tree, map_size, map_size)
                if accumulated_policy_values is None:
                    accumulated_policy_values = policy_values
                else:
                    accumulated_policy_values += policy_values
                if accumulated_counts is None:
                    accumulated_counts = counts
                else:
                    accumulated_counts += counts
        averaged_policy_values = np.divide(
            accumulated_policy_values,
            accumulated_counts,
            out=np.zeros_like(accumulated_policy_values),
            where=accumulated_counts != 0
        )
        # Also return the mask of explored states
        explored_mask = accumulated_counts != 0
        return averaged_policy_values, explored_mask

    # Compute averaged policy values and masks for both configs
    avg1, mask1 = get_averaged_policy_values(run_config1)
    avg2, mask2 = get_averaged_policy_values(run_config2)
    # Only show states explored in both runs
    both_explored = mask1 & mask2
    diff = avg1 - avg2
    diff = np.round(diff, 2)
    # Set unexplored states to np.nan so they are not visualized
    diff[~both_explored] = np.nan

    # Use the first config for plotting context
    hparams = run_config1
    map_size = hparams["map_size"]
    test_desc = hparams["test_env"]["desc"]
    obst_coords = []
    for i in range(len(test_desc)):
        for j in range(len(test_desc[0])):
            if test_desc[i][j] == "H":
                obst_coords.append((i, j))

    # Plot the difference
    diff_cmap = sns.diverging_palette(10, 120, as_cmap=True, center="light")
    ax = plot_density(diff, invert_fz_coordinate(root_state, map_size), obst_coords, map_size, map_size, cmap=diff_cmap)
    ax.set_title("Difference in Averaged Policy Values (Config1 - Config2)")
    plt.show()
    plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="AlphaZero Evaluation Configuration")
    map_size = 16

    TRAIN_CONFIG = "NO_HOLES" # NO_HOLES, MAZE_RL, MAZE_LR

    TEST_CONFIG = "SLALOM" # NO_OBSTACLES, HOLES, MAZE_RL, MAZE_LR

    parser.add_argument("--ENV", type=str, default="FROZENLAKE", help="Environment name")
    
    parser.add_argument("--map_size", type=int, default= map_size, help="Map size")
    parser.add_argument("--test_config", type=str, default= TEST_CONFIG, help="Config desc name")
    parser.add_argument("--train_config", type=str, default= TRAIN_CONFIG, help="Config desc name")

    # Run configurations
    parser.add_argument("--wandb_logs", type=bool, default= False, help="Enable wandb logging")
    parser.add_argument("--workers", type=int, default= 1, help="Number of workers")
    parser.add_argument("--runs", type=int, default= 1, help="Number of runs")

    # Basic search parameters
    parser.add_argument("--tree_evaluation_policy", type= str, default="mvc", help="Tree evaluation policy")
    parser.add_argument("--selection_policy", type=str, default="PolicyPUCT", help="Selection policy")
    parser.add_argument("--puct_c", type=float, default= 0, help="PUCT parameter")

    # Only relevant for single run evaluation
    parser.add_argument("--planning_budget", type=int, default = 128, help="Planning budget")
    # Only for MCTS
    parser.add_argument("--rollout_budget", type=int, default= 100, help="Rollout budget")

    # Search algorithm
    parser.add_argument("--agent_type", type=str, default= "azmcts_no_loops", help="Agent type")

    # Stochasticity parameters
    parser.add_argument("--eval_temp", type=float, default= 0, help="Temperature in tree evaluation softmax")
    parser.add_argument("--dir_epsilon", type=float, default= 0.0, help="Dirichlet noise parameter epsilon")
    parser.add_argument("--dir_alpha", type=float, default= None, help="Dirichlet noise parameter alpha")

    parser.add_argument("--tree_temperature", type=float, default= None, help="Temperature in tree evaluation softmax")

    parser.add_argument("--beta", type=float, default= 10, help="Beta parameter for mvc policy")

    # AZDetection detection parameters
    parser.add_argument("--threshold", type=float, default= 0.05, help="Detection threshold")
    parser.add_argument("--unroll_budget", type=int, default= 4, help="Unroll budget")

    # AZDetection replanning parameters
    parser.add_argument("--value_search", type=bool, default=True, help="Enable value search")
    parser.add_argument("--predictor", type=str, default="current_value", help="Predictor to use for detection")
    parser.add_argument("--update_estimator", type=bool, default=True, help="Update the estimator")

    # Test environment
    parser.add_argument("--test_env_is_slippery", type=bool, default= False, help="Slippery environment")
    parser.add_argument("--test_env_hole_reward", type=int, default=0, help="Hole reward")
    parser.add_argument("--test_env_terminate_on_hole", type=bool, default= False, help="Terminate on hole")
    parser.add_argument("--deviation_type", type=str, default= "bump", help="Deviation type")

    parser.add_argument("--ll_test_config", type=str, default= "TWO_LATERAL_PENTAGONS", help="LunarLander test config")

    # Model file
    parser.add_argument("--model_file", type=str, default= "", help="Model file")

    parser.add_argument( "--train_seeds", type=int, default=10, help="The number of random seeds to use for training.")
    parser.add_argument("--eval_seeds", type=int, default=1, help="The number of random seeds to use for evaluation.")

    # Rendering
    parser.add_argument("--render", type=bool, default=False, help="Render the environment")
    parser.add_argument("--visualize_trees", type=bool, default=True, help="Visualize trees")

    parser.add_argument("--hpc", type=bool, default=False, help="HPC flag")

    parser.add_argument("--value_estimate", type=str, default="nn", help="Value estimate method")

    parser.add_argument("--var_penalty", type=float, default=1, help="Variance penalty")
    parser.add_argument("--value_penalty", type=float, default=0, help="Value penalty")

    parser.add_argument("--final", type=bool, default=False)

    parser.add_argument("--save", type=bool, default=False)

    parser.add_argument("--bad_training", type=bool, default=False)

    parser.add_argument("--reuse_tree", type=bool, default=True, help="Update the estimator")
    parser.add_argument("--block_loops", type=bool, default=False, help="Block loops")

    parser.add_argument("--plot_tree_densities", type=bool, default=False, help="Plot tree densities")

    parser.add_argument("--max_episode_length", type=int, default=100, help="Max episode length")

    parser.add_argument("--discount_factor", type=float, default=0.95, help="Discount factor")

    parser.add_argument("--subopt_threshold", type=float, default=0.1, help="Suboptimality threshold")

    # Parse arguments
    args = parser.parse_args()

    single_train_seed = 1 # Only for single run
    single_eval_seed = 0 # Only for single run

    ENV = args.ENV

    args.test_env_id = f"CustomFrozenLakeNoHoles{args.map_size}x{args.map_size}-v1"
    args.test_env_desc = f"{args.map_size}x{args.map_size}_{args.test_config}"
    if args.map_size == 8 and args.train_config == "NO_HOLES":
        args.model_file = f"hyper/AZTrain_env=CustomFrozenLakeNoHoles8x8-v1_evalpol=visit_iterations=50_budget=64_df=0.95_lr=0.001_nstepslr=2_seed={single_train_seed}/checkpoint.pth"
    elif args.map_size == 16 and args.train_config == "NO_HOLES":
        args.model_file = f"hyper/AZTrain_env=CustomFrozenLakeNoHoles16x16-v1_evalpol=visit_iterations=60_budget=128_df=0.95_lr=0.003_nstepslr=2_seed={single_train_seed}/checkpoint.pth"
    elif args.map_size == 8 and args.train_config == "MAZE_RL":
        if args.bad_training:
            args.model_file = f"hyper/AZTrain_env=8x8_MAZE_RL_evalpol=visit_iterations=20_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={single_train_seed}/checkpoint.pth"
        else:
            args.model_file = f"hyper/AZTrain_env=8x8_MAZE_RL_evalpol=visit_iterations=150_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={single_train_seed}/checkpoint.pth"
    elif args.map_size == 8 and args.train_config == "MAZE_LR":
        if args.bad_training:
            args.model_file = f"hyper/AZTrain_env=8x8_MAZE_LR_evalpol=visit_iterations=10_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={single_train_seed}/checkpoint.pth"
        else:
            args.model_file = f"hyper/AZTrain_env=8x8_MAZE_LR_evalpol=visit_iterations=100_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={single_train_seed}/checkpoint.pth"
    elif args.map_size == 16 and args.train_config == "MAZE_LR":
        args.model_file = f"hyper/AZTrain_env=16x16_MAZE_LR_evalpol=visit_iterations=150_budget=64_df=0.95_lr=0.003_nstepslr=2_c=0.2_seed={single_train_seed}/checkpoint.pth"

    challenge = env_challenges[f"CustomFrozenLakeNoHoles{args.map_size}x{args.map_size}-v1"]  # Training environment

    observation_embedding = "coordinate"

    test_env_dict = {
        "id": args.test_env_id,
        "desc": fz_env_descriptions[args.test_env_desc],
        "is_slippery": args.test_env_is_slippery,
        "hole_reward": args.test_env_hole_reward,
        "terminate_on_hole": args.test_env_terminate_on_hole,
        "deviation_type": args.deviation_type,
    }  

    map_name = args.test_env_desc

    # Construct the config
    config_modifications = {
        "wandb_logs": args.wandb_logs,
        "workers": args.workers,
        "runs": args.runs,
        "tree_evaluation_policy": args.tree_evaluation_policy,
        "selection_policy": args.selection_policy,
        "planning_budget": args.planning_budget,
        "discount_factor": args.discount_factor,
        "puct_c": args.puct_c,
        "agent_type": args.agent_type,
        "eval_temp": args.eval_temp,
        "dir_epsilon": args.dir_epsilon,
        "dir_alpha": args.dir_alpha,
        "threshold": args.threshold,
        "unroll_budget": args.unroll_budget,
        "value_search": args.value_search,
        "predictor": args.predictor,
        "map_name": map_name,
        "test_env": test_env_dict,
        "observation_embedding": observation_embedding,
        "model_file": args.model_file,
        "render": args.render,
        "hpc": args.hpc,
        "value_estimate": args.value_estimate,
        "visualize_trees": args.visualize_trees,
        "map_size": args.map_size,
        "var_penalty": args.var_penalty,
        "value_penalty": args.value_penalty,
        "update_estimator": args.update_estimator,
        "train_config": args.train_config,
        "test_config": args.test_config,
        "eval_param": args.beta,
        "tree_temperature": args.tree_temperature,
        "save": args.save,
        "bad_training": args.bad_training,
        "reuse_tree": args.reuse_tree,
        "plot_tree_densities": args.plot_tree_densities,
        "max_episode_length": args.max_episode_length,
        "rollout_budget": args.rollout_budget,
        "subopt_threshold": args.subopt_threshold,
        "block_loops": args.block_loops,
    }

    run_config = {**base_parameters, **challenge, **config_modifications}

    ROOT_STATE = (7,14)

    eval_seeds = list(range(10))
    train_seeds = list(range(10))
    create_average_density_visualizations(run_config, ROOT_STATE, train_seeds, eval_seeds)

    # run_config2 = run_config.copy()

    # run_config2["value_penalty"] = 1

    # compare_average_density_visualizations(run_config2, run_config, ROOT_STATE, train_seeds, eval_seeds)




