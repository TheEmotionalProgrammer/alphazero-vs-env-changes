from pdb import run
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
import experiments.parameters as parameters
from environments.observation_embeddings import ObservationEmbedding, embedding_dict
from environments.frozenlake.fz_configs import *
from az.azmcts import AlphaZeroMCTS, AlphaZeroMCTS_T
from azdetection.change_detector import AlphaZeroDetector
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


def agent_from_config(hparams: dict):

    env = gym.make(**hparams["env_params"])

    discount_factor = hparams["discount_factor"]

    if "tree_temperature" not in hparams:
        hparams["tree_temperature"] = None

    if "tree_value_transform" not in hparams or hparams["tree_value_transform"] is None:
        hparams["tree_value_transform"] = "identity"

    tree_evaluation_policy = tree_eval_dict(
        hparams["eval_param"],
        discount_factor,
        hparams["puct_c"],
        hparams["tree_temperature"],
        value_transform=value_transform_dict[hparams["tree_value_transform"]],
    )[hparams["tree_evaluation_policy"]]
    if (
        "selection_value_transform" not in hparams
        or hparams["selection_value_transform"] is None
    ):
        hparams["selection_value_transform"] = "identity"

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
    if "observation_embedding" not in hparams:
        hparams["observation_embedding"] = "default"

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

        if "dir_epsilon" not in hparams:
            hparams["dir_epsilon"] = 0.0
            hparams["dir_alpha"] = None

        print("Direpslion: ", hparams["dir_epsilon"])

        dir_epsilon = hparams["dir_epsilon"]
        dir_alpha = hparams["dir_alpha"]

        threshold = hparams["threshold"]

        planning_style = hparams["planning_style"]

        value_search = hparams["value_search"]

        predictor = hparams["predictor"]

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
    workers = hparams["workers"]

    seeds = [None] * hparams["runs"]

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
    )
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

def eval_single():

    register_all() # Register custom environments that we are going to use

    challenge = parameters.env_challenges[3] # Training environment

    config_modifications = {
        
        # Run configurations
        "wandb_logs": False,
        "workers": min(6, multiprocessing.cpu_count()),
        "runs": 1,

        # Basic search parameters
        "tree_evaluation_policy": "mvc",
        "selection_policy": "PolicyUCT",
        "planning_budget": 64,
        "puct_c": 0.0,

        # Search algorithm
        "agent_type": "azdetection", 
        "depth_estimation": False,

        # Stochasticity parameters
        "eval_temp": 0, # Temperature in tree evaluation softmax, 0 means we are taking the stochastic argmax of the distribution
        "dir_epsilon": 0.0, # Dirichlet noise parameter
        "dir_alpha": None, # Dirichlet noise parameter

        # AZDetection detection parameters
        "threshold": 0.03, # NOTE: this is going to be ignored if the predictor is original_env
        "unroll_budget": 10, 

        # AZDetection replanning parameters
        "planning_style": "connected",
        "value_search": True,
        "predictor": "current_value", # The predictor to use for the detection

        # Test environment with obstacles position specified in desc
        "test_env": dict(    
            id = "DefaultFrozenLake8x8-v1",
            desc = NARROW,
            is_slippery=False,
            hole_reward=0,
            terminate_on_hole=False,
           
        ),
        "observation_embedding": "coordinate", # When the observation is just a coordinate on a grid, can use coordinate

        "model_file": "/Users/isidorotamassia/THESIS/alphazero-vs-env-changes/runs/hyper/CustomFrozenLakeNoHoles8x8-v1_20241216-003012/checkpoint.pth",
    }

    run_config = {**parameters.base_parameters, **challenge, **config_modifications}

    return eval_from_config(config=run_config)

def eval_budget_sweep(
    project_name="AlphaZeroEval",
    entity=None,
    config=None,
    budgets=None,
    num_seeds=None,
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
        run_name = f"Algorithm_({config['agent_type']})_EvalPol_({config['tree_evaluation_policy']})_SelPol_({config['selection_policy']})_Predictor_({config['predictor']})_PlanningStyle_({config['planning_style']})_ValueSearch_({config['value_search']})"
    elif config["agent_type"] == "azmcts":
        run_name = f"Algorithm_({config['agent_type']})_EvalPol_({config['tree_evaluation_policy']})_SelPol_({config['selection_policy']})"

    if budgets is None:
        budgets = [
            8, 16, 32, 64, 128             
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
    
    for seed in range(num_seeds):

        for budget in budgets:
            # Make a copy of the base configuration to avoid modifying the original
            config_copy = dict(hparams)
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
                wandb.log({"Planning_Budget": budget, f"Mean_Discounted_Return_{seed}": mean_discounted_return})
                wandb.log({"Planning_Budget": budget, f"Mean_Return_{seed}": mean_return})
                wandb.log({"Planning_Budget": budget, f"Mean_Episode_Length_{seed}": mean_time_steps})

        # # Print or log the final results for discounted return, return and episode length
        # print("Budget Sweep Results:")
        # for budget, mean_discounted_return, mean_return, mean_time_steps in budget_results:
        #     print(f"Planning Budget: {budget}, Mean Discounted Return: {mean_discounted_return}, Mean Return: {mean_return}, Mean Episode Length: {mean_time_steps}")
        
    if use_wandb:
        run.finish()

if __name__ == "__main__":

    challenge = parameters.env_challenges[3] # Training environment

    config_modifications = {
        
        # Run configurations
        "wandb_logs": True,
        "workers": 6,
        "runs": 1,

        # Basic search parameters
        "tree_evaluation_policy": "visit",
        "selection_policy": "UCT",
        "planning_budget": 64,
        #"puct_c": 0.0,

        # Search algorithm
        "agent_type": "azmcts", # Classic azmcts or novel azdetection
        "depth_estimation": False, # Whether to use tree depth estimation by Moerland et al.

        # Stochasticity parameters
        "eval_temp": 0, # Temperature in tree evaluation softmax, 0 means we are taking the stochastic argmax of the distribution
        "dir_epsilon": 0.0, # Dirichlet noise parameter
        "dir_alpha": None, # Dirichlet noise parameter

        # AZDetection detection parameters
        "threshold": 0.03, # NOTE: this is going to be ignored if the predictor is original_env
        "unroll_budget": 10, 

        # AZDetection replanning parameters
        "planning_style": "mini_trees",
        "value_search": True,
        "predictor": "current_value", # The predictor to use for the detection

        # Test environment with obstacles position specified in desc
        "test_env": dict(    
            id = "DefaultFrozenLake8x8-v1",
            desc = NARROW,
            is_slippery=False,
            hole_reward=0,
            terminate_on_hole=False,
           
        ),
        "observation_embedding": "coordinate", # When the observation is just a coordinate on a grid, can use coordinate

        "model_file": "/scratch/itamassia/alphazero-vs-env-changes/hyper/AZTrain_env=CustomFrozenLakeNoHoles8x8-v1_iterations=50_budget=64_seed=1_20241224-180758/checkpoint.pth",
    }

    run_config = {**parameters.base_parameters, **challenge, **config_modifications}
    # sweep_id = wandb.sweep(sweep=coord_search, project="AlphaZero")

    # wandb.agent(sweep_id, function=sweep_agent)
    eval_budget_sweep(config=run_config, num_seeds = 50)
