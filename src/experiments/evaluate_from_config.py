import sys

sys.path.append("src/")
import time

from tqdm import tqdm
import numpy as np
import multiprocessing
import gymnasium as gym
import wandb

from log_code.metrics import calc_metrics
from experiments.eval_agent import eval_agent
from core.mcts import DistanceMCTS, LakeDistanceMCTS, RandomRolloutMCTS
from environments.observation_embeddings import ObservationEmbedding, embedding_dict
from az.azmcts import AlphaZeroMCTS, AlphaZeroMCTS_T
from azdetection.change_detector import AlphaZeroDetector, AlphaZeroDetector_T
from az.model import (
    AlphaZeroModel,
    models_dict
)
from policies.tree_policies import tree_eval_dict
from policies.selection_distributions import selection_dict_fn
from policies.value_transforms import value_transform_dict
import torch as th

import copy
from environments.register import register_all

import argparse
from parameters import base_parameters, env_challenges, fz_env_descriptions

from log_code.gen_renderings import save_gif_imageio
from log_code.tree_visualizer import visualize_trees

def agent_from_config(hparams: dict):
    
    env = gym.make(**hparams["env_params"])

    discount_factor = hparams["discount_factor"]

    tree_evaluation_policy = tree_eval_dict(
        hparams["eval_param"],
        discount_factor,
        hparams["puct_c"],
        hparams["tree_temperature"],
        value_transform=value_transform_dict[hparams["tree_value_transform"]],
    )[hparams["tree_evaluation_policy"]]

    selection_policy = selection_dict_fn(
        hparams["puct_c"],
        tree_evaluation_policy,
        discount_factor,
        value_transform_dict[hparams["selection_value_transform"]],
    )[hparams["selection_policy"]]

    if (
        "root_selection_policy" not in hparams
        or hparams["root_selection_policy"] is None
    ):
        hparams["root_selection_policy"] = hparams["selection_policy"]


    root_selection_policy = selection_dict_fn(
        hparams["puct_c"],
        tree_evaluation_policy,
        discount_factor,
        value_transform_dict[hparams["selection_value_transform"]],
    )[hparams["root_selection_policy"]]

    if "estimation_policy" not in hparams or hparams["estimation_policy"] is None:
        hparams["estimation_policy"] = "UCT"

    estimation_policy = selection_dict_fn(
        hparams["puct_c"],
        tree_evaluation_policy,
        discount_factor,
        value_transform_dict[hparams["selection_value_transform"]],
    )[hparams["estimation_policy"]]

    observation_embedding: ObservationEmbedding = embedding_dict[
        hparams["observation_embedding"]
    ](env.observation_space, hparams["ncols"] if "ncols" in hparams else None)

    if hparams["agent_type"] == "random_rollout":
        if "rollout_budget" not in hparams:
            hparams["rollout_budget"] = 40
        agent = RandomRolloutMCTS(
            rollout_budget=hparams["rollout_budget"],
            root_selection_policy=root_selection_policy,
            selection_policy=selection_policy,
            discount_factor=discount_factor,
        )

    elif hparams["agent_type"] == "distance":
        if "desc" in hparams["env_params"] and hparams["env_params"]["desc"] is not None:
            lake_map = hparams["env_params"]["desc"]
            agent = LakeDistanceMCTS(
                lake_map=lake_map,
                root_selection_policy=root_selection_policy,
                selection_policy=selection_policy,
                discount_factor=discount_factor,
            )
        else:

            agent = DistanceMCTS(
                embedding=observation_embedding,
                root_selection_policy=root_selection_policy,
                selection_policy=selection_policy,
                discount_factor=discount_factor,
            )

    elif hparams["agent_type"] == "azmcts":

        filename = hparams["model_file"]

        model: AlphaZeroModel = models_dict[hparams["model_type"]].load_model(
            filename, 
            env, 
            False, 
            hparams["hidden_dim"]

        )

        model.eval()

        if "dir_epsilon" not in hparams:
            hparams["dir_epsilon"] = 0.0
            hparams["dir_alpha"] = None

        dir_epsilon = hparams["dir_epsilon"]
        dir_alpha = hparams["dir_alpha"]

        if hparams["depth_estimation"]:
            agent = AlphaZeroMCTS_T(
                model=model,
                selection_policy=selection_policy,
                discount_factor=discount_factor,
                root_selection_policy=root_selection_policy,
                estimation_policy=estimation_policy,
                dir_epsilon=dir_epsilon,
                dir_alpha=dir_alpha,
            )
        else:
            agent = AlphaZeroMCTS(
                root_selection_policy=root_selection_policy,
                selection_policy=selection_policy,
                model=model,
                dir_epsilon=dir_epsilon,
                dir_alpha=dir_alpha,
                discount_factor=discount_factor,
                value_estimate=hparams["value_estimate"],
            )

    elif hparams["agent_type"] == "azdetection":

        filename = hparams["model_file"]

        model: AlphaZeroModel = models_dict[hparams["model_type"]].load_model(
            filename, 
            env, 
            False, 
            hparams["hidden_dim"]
        )

        model.eval()

        print("Direpslion: ", hparams["dir_epsilon"])

        dir_epsilon = hparams["dir_epsilon"]
        dir_alpha = hparams["dir_alpha"]

        threshold = hparams["threshold"]

        planning_style = hparams["planning_style"]

        value_search = hparams["value_search"]

        predictor = hparams["predictor"]

        if hparams["depth_estimation"]:

            agent = AlphaZeroDetector_T(
                predictor=predictor,
                root_selection_policy=root_selection_policy,
                selection_policy=selection_policy,
                model=model,
                dir_epsilon=dir_epsilon,
                dir_alpha=dir_alpha,
                discount_factor=discount_factor,
                threshold=threshold,
                planning_style=planning_style,
                value_search=value_search,
                estimation_policy=estimation_policy
            )

        else:
            agent = AlphaZeroDetector(
                predictor=predictor,
                root_selection_policy=root_selection_policy,
                selection_policy=selection_policy,
                model=model,
                dir_epsilon=dir_epsilon,
                dir_alpha=dir_alpha,
                discount_factor=discount_factor,
                threshold=threshold,
                planning_style=planning_style,
                value_search=value_search,
                value_estimate=hparams["value_estimate"],
            )

    else:

        raise ValueError(f"Unknown agent type {hparams['agent_type']}")

        
    return (
        agent,
        env,
        tree_evaluation_policy,
        observation_embedding,
        hparams["planning_budget"],
    )


def eval_from_config(
    project_name="AlphaZero", entity=None, job_name=None, config=None, tags=None
):
    if tags is None:
        tags = []
    tags.append("evaluation")

    use_wandb = config["wandb_logs"]

    if use_wandb:
        # Initialize Weights & Biases
        settings = wandb.Settings(job_name=job_name)
        run = wandb.init(
            project=project_name, entity=entity, settings=settings, config=config, tags=tags
        )
        assert run is not None
        hparams = wandb.config
    else:
        hparams = config

    agent, train_env, tree_evaluation_policy, observation_embedding, planning_budget = (
        agent_from_config(hparams)
    )

    if "workers" not in hparams or hparams["workers"] is None:
        hparams["workers"] = multiprocessing.cpu_count()
    else:
        workers = hparams["workers"]

    seeds = [0] * hparams["runs"]

    test_env = gym.make(**hparams["test_env"])

    results = eval_agent(
        agent=agent,
        env=test_env,
        original_env=train_env,
        tree_evaluation_policy=tree_evaluation_policy,
        observation_embedding=observation_embedding,
        planning_budget=planning_budget,
        max_episode_length=hparams["max_episode_length"],
        seeds=seeds,
        temperature=hparams["eval_temp"],
        workers=workers,
        azdetection= (hparams["agent_type"] == "azdetection"),
        unroll_budget= hparams["unroll_budget"],
        render=hparams["render"],
        return_trees=hparams["visualize_trees"],
    )

    if hparams["visualize_trees"]:
        results, trees = results
        trees = trees[0]
        print(f"Visualizing {len(trees)} trees...")
        visualize_trees(trees, "tree_visualizations")
        
    episode_returns, discounted_returns, time_steps, entropies = calc_metrics(
        results, agent.discount_factor, test_env.action_space.n
    )

    trajectories = []
    for i in range(results.shape[0]):
        re = []
        for j in range(results.shape[1]):
            re.append(
                observation_embedding.tensor_to_obs(results[i, j]["observations"])
            )
            if results[i, j]["terminals"] == 1:
                break
        trajectories.append(re)

    eval_res = {
        # wandb logs
        "Evaluation/Returns": wandb.Histogram(np.array((episode_returns))),
        "Evaluation/Discounted_Returns": wandb.Histogram(np.array((discounted_returns))),
        "Evaluation/Timesteps": wandb.Histogram(np.array((time_steps))),
        # "Evaluation/Entropies": wandb.Histogram(np.array(((th.sum(entropies, dim=-1) / time_steps)))),

        # standard logs
        "Evaluation/Mean_Returns": episode_returns.mean().item(),
        "Evaluation/Mean_Discounted_Returns": discounted_returns.mean().item(),
        # "Evaluation/Mean_Entropy": (th.sum(entropies, dim=-1) / time_steps).mean().item(),
        "trajectories": trajectories,
    }

    if use_wandb:
        run.log(data=eval_res)
        run.log_code(root="./src")
        # Finish the WandB run
        run.finish()
    
    else:
        print(f"Evaluation Mean Return: {eval_res['Evaluation/Mean_Returns']}")
        print(f"Evaluation Mean Discounted Return: {eval_res['Evaluation/Mean_Discounted_Returns']}")

def eval_budget_sweep(
    project_name="AlphaZeroEval",
    entity=None,
    config=None,
    budgets=None,
    num_train_seeds=None,
    num_eval_seeds=None,
):
    """
    Evaluate the agent with increasing planning budgets and log the results.

    Args:
        project_name (str): WandB project name.
        entity (str): WandB entity name.
        job_name (str): Job name for WandB logs.
        config (dict): Base configuration for the agent.
        budgets (list): List of planning budgets to evaluate.
        num_seeds (int): Number of seeds to run.
    """
    if config["agent_type"] == "azdetection":
        run_name = f"Algorithm_({config['agent_type']})_EvalPol_({config['tree_evaluation_policy']})_SelPol_({config['selection_policy']})_Predictor_({config['predictor']})_n_({config['unroll_budget']})_eps_({config['threshold']})_PlanningStyle_({config['planning_style']})_ValueSearch_({config['value_search']})_{config['map_name']}"
    elif config["agent_type"] == "azmcts":
        run_name = f"Algorithm_({config['agent_type']})_EvalPol_({config['tree_evaluation_policy']})_SelPol_({config['selection_policy']})_{config['map_name']}"

    if budgets is None:
        budgets = [
            64#8, 16, 32, 64, 128             
        ]  # Default budgets to sweep

    use_wandb = config["wandb_logs"]
    
    if use_wandb:
        # Initialize WandB run
        run = wandb.init(
            project=project_name, 
            entity=entity, 
            name=run_name, 
            config=config, 
            tags=["budget_sweep"]
        )
        hparams = wandb.config
    else:
        hparams = config

    # Register custom environments
    register_all()

    # Store results for plotting
    budget_results = []

    for model_seed in range(num_train_seeds):

        model_file = f"{"hyper" if not hparams["hpc"] else "scratch/itamassia"}/AZTrain_env=CustomFrozenLakeNoHoles8x8-v1_iterations=50_budget=64_seed={model_seed}/checkpoint.pth"

        for seed in range(num_eval_seeds):

            for budget in budgets:
                # Make a copy of the base configuration to avoid modifying the original
                config_copy = dict(hparams)

                config_copy["model_file"] = model_file
                config_copy["planning_budget"] = budget
                
                print(f"Running evaluation for planning_budget={budget}")

                # Evaluate the agent
                agent, train_env, tree_evaluation_policy, observation_embedding, planning_budget = agent_from_config(config_copy)
                test_env = gym.make(**config_copy["test_env"])
                #seeds = [None] * config_copy["runs"]
                seeds = [seed]
                
                results = eval_agent(
                    agent=agent,
                    env=test_env,
                    original_env=train_env,
                    tree_evaluation_policy=tree_evaluation_policy,
                    observation_embedding=observation_embedding,
                    planning_budget=planning_budget,
                    max_episode_length=config_copy["max_episode_length"],
                    seeds=seeds,
                    temperature=config_copy["eval_temp"],
                    workers=config_copy["workers"],
                    azdetection=(config_copy["agent_type"] == "azdetection"),
                    unroll_budget=config_copy["unroll_budget"],
                )

                # Calculate metrics
                episode_returns, discounted_returns, time_steps, _ = calc_metrics(
                    results, agent.discount_factor, test_env.action_space.n
                )

                # Compute mean discounted return
                mean_discounted_return = discounted_returns.mean().item()
                mean_return = episode_returns.mean().item()
                mean_time_steps = time_steps.mean().item()
                budget_results.append((budget, mean_discounted_return, mean_return, mean_time_steps))

                if use_wandb:
                    wandb.log({"Planning_Budget": budget, f"Mean_Discounted_Return_trainseed={model_seed}_evalseed={seed}": mean_discounted_return})
                    wandb.log({"Planning_Budget": budget, f"Mean_Return_trainseed={model_seed}_evalseed={seed}": mean_return})
                    wandb.log({"Planning_Budget": budget, f"Mean_Episode_Length_trainseed={model_seed}_evalseed={seed}": mean_time_steps})

                else:
                    print(f"Mean Discounted Return: {mean_discounted_return}")
                    print(f"Mean Return: {mean_return}")
                    print(f"Mean Episode Length: {mean_time_steps}")
        
    if use_wandb:
        run.finish()
    else:
        # Print the mean of the results for each budget
        for budget in budgets:
            budget_results = np.array(budget_results)
            mean_discounted_return = np.mean(budget_results[budget_results[:, 0] == budget, 1])
            mean_return = np.mean(budget_results[budget_results[:, 0] == budget, 2])
            mean_time_steps = np.mean(budget_results[budget_results[:, 0] == budget, 3])
            print(f"Planning Budget: {budget}")
            print(f"Mean Discounted Return: {mean_discounted_return}")
            print(f"Mean Return: {mean_return}")
            print(f"Mean Episode Length: {mean_time_steps}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="AlphaZero Evaluation Configuration")

    # Run configurations
    parser.add_argument("--wandb_logs", type=bool, default=False, help="Enable wandb logging")
    parser.add_argument("--wandb_logs", type=bool, default=False, help="Enable wandb logging")
    parser.add_argument("--workers", type=int, default=6, help="Number of workers")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")

    # Basic search parameters
    parser.add_argument("--tree_evaluation_policy", type=str, default="visit", help="Tree evaluation policy")
    parser.add_argument("--selection_policy", type=str, default="PUCT", help="Selection policy")
    parser.add_argument("--planning_budget", type=int, default=32, help="Planning budget")
    parser.add_argument("--puct_c", type=float, default=1.0, help="PUCT parameter")

    # Search algorithm
    parser.add_argument("--agent_type", type=str, default="azdetection", help="Agent type")
    parser.add_argument("--depth_estimation", type=bool, default=False, help="Use tree depth estimation")

    # Stochasticity parameters
    parser.add_argument("--eval_temp", type=float, default=0.0, help="Temperature in tree evaluation softmax")
    parser.add_argument("--dir_epsilon", type=float, default=0.0, help="Dirichlet noise parameter epsilon")
    parser.add_argument("--dir_alpha", type=float, default=None, help="Dirichlet noise parameter alpha")

    # AZDetection detection parameters
    parser.add_argument("--threshold", type=float, default=0.01, help="Detection threshold")
    parser.add_argument("--unroll_budget", type=int, default=10, help="Unroll budget")

    # AZDetection replanning parameters
    parser.add_argument("--planning_style", type=str, default="mini_trees", help="Planning style")
    parser.add_argument("--value_search", type=bool, default=True, help="Enable value search")
    parser.add_argument("--predictor", type=str, default="original_env", help="Predictor to use for detection")

    # Test environment
    parser.add_argument("--test_env_id", type=str, default="CustomFrozenLakeNoHoles16x16-v1", help="Test environment ID")
    parser.add_argument("--test_env_desc", type=str, default="16x16_IMPOSSIBLE", help="Environment description")
    parser.add_argument("--test_env_is_slippery", type=bool, default=False, help="Environment slippery flag")
    parser.add_argument("--test_env_hole_reward", type=int, default=0, help="Hole reward")
    parser.add_argument("--test_env_terminate_on_hole", type=bool, default=False, help="Terminate on hole")
    parser.add_argument("--deviation_type", type=str, default="bump", help="Deviation type")

    # Observation embedding
    parser.add_argument("--observation_embedding", type=str, default="coordinate", help="Observation embedding type")

    # Model file for single run evaluation
    parser.add_argument("--model_file", type=str, default=f"hyper/AZTrain_env=CustomFrozenLakeNoHoles16x16-v1_iterations=200_budget=32_seed=7_20250121-104757/checkpoint.pth", help="Path to model file")

    parser.add_argument("--train_seeds", type=int, default=10, help="The number of random seeds to use for training.")
    parser.add_argument("--eval_seeds", type=int, default=1, help="The number of random seeds to use for evaluation.")

    # Rendering
    parser.add_argument("--render", type=bool, default=True, help="Render the environment")

    parser.add_argument("--run_full_eval", type=bool, default= False, help="Run type")

    parser.add_argument("--hpc", type=bool, default=False, help="HPC flag")

    parser.add_argument("--value_estimate", type=str, default="nn", help="Value estimate method")

    # Parse arguments
    args = parser.parse_args()

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
        "depth_estimation": args.depth_estimation,
        "eval_temp": args.eval_temp,
        "dir_epsilon": args.dir_epsilon,
        "dir_alpha": args.dir_alpha,
        "threshold": args.threshold,
        "unroll_budget": args.unroll_budget,
        "planning_style": args.planning_style,
        "value_search": args.value_search,
        "predictor": args.predictor,
        "map_name": args.test_env_desc,
        "test_env": {
            "id": args.test_env_id,
            "desc": fz_env_descriptions[args.test_env_desc],
            "is_slippery": args.test_env_is_slippery,
            "hole_reward": args.test_env_hole_reward,
            "terminate_on_hole": args.test_env_terminate_on_hole,
            "deviation_type": args.deviation_type,
        },
        "observation_embedding": args.observation_embedding,
        "model_file": args.model_file,
        "render": args.render,
        "hpc": args.hpc,
        "value_estimate": args.value_estimate,
    }

    run_config = {**base_parameters, **challenge, **config_modifications}

    # Execute the evaluation

    if args.run_full_eval:
        eval_budget_sweep(config=run_config, num_train_seeds=args.train_seeds, num_eval_seeds=args.eval_seeds)
    else:
        eval_from_config(config=run_config)
