import sys

sys.path.append("src/")

import os
import numpy as np
import multiprocessing
import gymnasium as gym
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from log_code.metrics import calc_metrics
from experiments.evaluation.eval_agent import eval_agent
from core.mcts import DistanceMCTS, LakeDistanceMCTS, RandomRolloutMCTS
from environments.observation_embeddings import ObservationEmbedding, embedding_dict
from az.azmcts import AlphaZeroMCTS

from azdetection.minitrees import MiniTrees
from azdetection.megatree import MegaTree
from azdetection.octopus import Octopus

from az.model import (
    AlphaZeroModel,
    models_dict
)

from policies.tree_policies import tree_eval_dict
from policies.selection_distributions import selection_dict_fn

from policies.value_transforms import value_transform_dict

from environments.register import register_all

import argparse
from experiments.parameters import base_parameters, env_challenges, fz_env_descriptions

from log_code.tree_visualizer import visualize_trees
from log_code.plot_state_densities import plot_density, calculate_density, calculate_nn_value_means, calculate_policy_value_means, calculate_variance_means

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

    #print("evaltemp",tree_evaluation_policy.temperature)

    selection_policy = selection_dict_fn(
        hparams["puct_c"],
        tree_evaluation_policy,
        discount_factor,
        value_transform=value_transform_dict[hparams["selection_value_transform"]],
    )[hparams["selection_policy"]]

    #print("seltemp",selection_policy.temperature)

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


        agent = AlphaZeroMCTS(
            root_selection_policy=root_selection_policy,
            selection_policy=selection_policy,
            model=model,
            dir_epsilon=dir_epsilon,
            dir_alpha=dir_alpha,
            discount_factor=discount_factor,
            value_estimate=hparams["value_estimate"],
        )

    elif hparams["agent_type"] == "mini-trees" or hparams["agent_type"] == "mega-tree" or hparams["agent_type"] == "octopus":

        filename = hparams["model_file"]

        model: AlphaZeroModel = models_dict[hparams["model_type"]].load_model(
            filename, 
            env, 
            False, 
            hparams["hidden_dim"]
        )

        model.eval()

        #print("Direpslion: ", hparams["dir_epsilon"])

        dir_epsilon = hparams["dir_epsilon"]
        dir_alpha = hparams["dir_alpha"]
        
        threshold = hparams["threshold"]

        value_search = hparams["value_search"]

        predictor = hparams["predictor"]

        if hparams["agent_type"] == "mega-tree":
            agent = MegaTree(
                predictor=predictor,
                root_selection_policy=root_selection_policy,
                selection_policy=selection_policy,
                model=model,
                dir_epsilon=dir_epsilon,
                dir_alpha=dir_alpha,
                discount_factor=discount_factor,
                threshold=threshold,
                value_search=value_search,
                value_estimate=hparams["value_estimate"],
                var_penalty=hparams["var_penalty"],
                value_penalty=hparams["value_penalty"],
                update_estimator=hparams["update_estimator"],
            )
        elif hparams["agent_type"] == "mini-trees":

            agent = MiniTrees(
                predictor=predictor,
                root_selection_policy=root_selection_policy,
                selection_policy=selection_policy,
                model=model,
                dir_epsilon=dir_epsilon,
                dir_alpha=dir_alpha,
                discount_factor=discount_factor,
                threshold=threshold,
                value_search=value_search,
                value_estimate=hparams["value_estimate"],
                update_estimator=hparams["update_estimator"],
            )

        elif hparams["agent_type"] == "octopus":
            agent = Octopus(
                predictor=predictor,
                root_selection_policy=root_selection_policy,
                selection_policy=selection_policy,
                model=model,
                dir_epsilon=dir_epsilon,
                dir_alpha=dir_alpha,
                discount_factor=discount_factor,
                threshold=threshold,
                value_estimate=hparams["value_estimate"],
                update_estimator=hparams["update_estimator"],
                var_penalty=hparams["var_penalty"],
                value_penalty=hparams["value_penalty"],
                policy_det_rule=hparams["policy_det_rule"],
                reuse_tree=hparams["reuse_tree"],
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

    seeds = [3] * hparams["runs"]

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
        azdetection= (hparams["agent_type"] == "mini-trees" or hparams["agent_type"] == "mega-tree"),
        unroll_budget= hparams["unroll_budget"],
        render=hparams["render"],
        return_trees=hparams["visualize_trees"] or hparams["plot_tree_densities"],
    )

    if hparams["visualize_trees"]:
        results, trees = results
        trees = trees[0]
        print(f"Visualizing {len(trees)} trees...")
        visualize_trees(trees, "tree_visualizations")

    if hparams["plot_tree_densities"]:
        results, trees = results
        trees = trees[0]

        test_desc = hparams["test_env"]["desc"]

        obst_coords = []
        for i in range(len(test_desc)):
            for j in range(len(test_desc[0])):
                if test_desc[i][j] == "H":
                    obst_coords.append((i, j))
        
        for i, tree in enumerate(trees):

            states_density = calculate_density(tree, len(test_desc[0]), len(test_desc))
            policy_values = calculate_policy_value_means(tree, len(test_desc[0]), len(test_desc))
            nn_values = calculate_nn_value_means(tree, len(test_desc[0]), len(test_desc))
            variances = calculate_variance_means(tree, len(test_desc[0]), len(test_desc))

            states_cmap = sns.diverging_palette(10, 120, as_cmap=True, center="light")
            values_cmap = sns.diverging_palette(120, 10, as_cmap=True, center="light")
            nn_values_cmap = sns.diverging_palette(120, 10, as_cmap=True, center="light")
            variances_cmap = sns.diverging_palette(20, 240, as_cmap=True, center="light")

            ax = plot_density(states_density, tree.observation, obst_coords, len(test_desc[0]), len(test_desc), cmap=states_cmap)

            ax.set_title(f"State Visitation Counts at step {i}")

            # If no path to folder, create it
            if not os.path.exists("states_density"):
                os.makedirs("states_density")

            plt.savefig(f"states_density/{i}.png")
            plt.close()

            ax = plot_density(policy_values, tree.observation, obst_coords, len(test_desc[0]), len(test_desc), cmap=values_cmap)

            ax.set_title(f"Mean Policy Values at step {i}")

            # If no path to folder, create it
            if not os.path.exists("policy_values"):
                os.makedirs("policy_values")

            plt.savefig(f"policy_values/{i}.png")
            plt.close()

            ax = plot_density(variances, tree.observation, obst_coords, len(test_desc[0]), len(test_desc), cmap=variances_cmap)

            ax.set_title(f"Mean Variances at step {i}")

            # If no path to folder, create it
            if not os.path.exists("variances"):
                os.makedirs("variances")

            plt.savefig(f"variances/{i}.png")
            plt.close()

            ax = plot_density(nn_values, tree.observation, obst_coords, len(test_desc[0]), len(test_desc), cmap=nn_values_cmap)

            ax.set_title(f"Mean NN Values at step {i}")

            # If no path to folder, create it
            if not os.path.exists("nn_values"):
                os.makedirs("nn_values")

            plt.savefig(f"nn_values/{i}.png")
            plt.close()
            
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
    final = False,
    save = False
):
    """
    Evaluate the agent with increasing planning budgets and log the results.

    Args:
        project_name (str): WandB project name.
        entity (str): WandB entity name.
        config (dict): Base configuration for the agent.
        budgets (list): List of planning budgets to evaluate.
        num_train_seeds (int): Number of training seeds.
        num_eval_seeds (int): Number of evaluation seeds.
    """
    if config["agent_type"] == "mini-trees" or config["agent_type"] == "mega-tree":
        run_name = f"Algorithm_({config['agent_type']})_EvalPol_({config['tree_evaluation_policy']})_SelPol_({config['selection_policy']})_c_({config['puct_c']})_Predictor_({config['predictor']})_n_({config['unroll_budget']})_eps_({config['threshold']})_ValueSearch_({config['value_search']})_ValueEst_({config['value_estimate']})_UpdateEst_({config['update_estimator']})_{config['map_size']}x{config['map_size']}_{config['train_config']}_{config['test_config']}"
    elif config["agent_type"] == "azmcts":
        run_name = f"Algorithm_({config['agent_type']})_EvalPol_({config['tree_evaluation_policy']})_SelPol_({config['selection_policy']})_c_({config['puct_c']})_ValueEst_({config['value_estimate']})_{config['map_size']}x{config['map_size']}_{config['train_config']}_{config['test_config']}"
    elif config["agent_type"] == "octopus":
        run_name = f"Algorithm_({config['agent_type']})_EvalPol_({config['tree_evaluation_policy']})_SelPol_({config['selection_policy']})_c_({config['puct_c']})_Predictor_({config['predictor']})_eps_({config['threshold']})_ValueEst_({config['value_estimate']})_({config['update_estimator']})_ttemp_({config['tree_temperature']})_Value_Penalty_{config['value_penalty']}_{config['map_size']}x{config['map_size']}_{config['train_config']}_{config['test_config']}"

    if config["bad_training"]:
        run_name = run_name + "_BAD"

    if config["agent_type"] == "octopus":
        if not config["reuse_tree"]:
            run_name = run_name + "_NO_REUSE"
        if config["value_penalty"] == 0:
            run_name = run_name + "_NO_VP"
        if config["tree_temperature"] is None:
            run_name = run_name + "_NONE_TEMP"
        if config["puct_c"] > 0:
            run_name = run_name + "_C>0"
    
    if config["test_env"]["deviation_type"] == "clockwise":
        run_name = "CW_" + run_name 
    
    if config["test_env"]["deviation_type"] == "counter_clockwise":
        run_name = "CCW_" + run_name
    
    print(f"Run Name: {run_name}")

    if budgets is None:
        budgets = [8, 16, 32, 64, 128]  # Default budgets to sweep

    use_wandb = config["wandb_logs"]

    if use_wandb:
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

    register_all()  # Register custom environments

    # Store results for plotting
    results_data = []

    for model_seed in range(num_train_seeds):
        print(f"Training Seed: {model_seed}")

        if config["map_size"] == 8 and config["train_config"] == "NO_HOLES":
            model_file = f"hyper/AZTrain_env=CustomFrozenLakeNoHoles8x8-v1_evalpol=visit_iterations=50_budget=64_df=0.95_lr=0.001_nstepslr=2_seed={model_seed}/checkpoint.pth"
        elif config["map_size"] == 16 and config["train_config"] == "NO_HOLES":
            model_file = f"hyper/AZTrain_env=CustomFrozenLakeNoHoles16x16-v1_evalpol=visit_iterations=60_budget=128_df=0.95_lr=0.003_nstepslr=2_seed={model_seed}/checkpoint.pth"

        elif config["map_size"] == 8 and config["train_config"] == "MAZE_RL":
            if config["bad_training"]:
                model_file = f"hyper/AZTrain_env=8x8_MAZE_RL_evalpol=visit_iterations=20_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={model_seed}/checkpoint.pth"
            else:
                model_file = f"hyper/AZTrain_env=8x8_MAZE_RL_evalpol=visit_iterations=150_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={model_seed}/checkpoint.pth"

        elif config["map_size"] == 8 and config["train_config"] == "MAZE_LR":
            if config["bad_training"]:
                model_file = f"hyper/AZTrain_env=8x8_MAZE_LR_evalpol=visit_iterations=10_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={model_seed}/checkpoint.pth"
            else:
                model_file = f"hyper/AZTrain_env=8x8_MAZE_LR_evalpol=visit_iterations=100_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={model_seed}/checkpoint.pth"

        elif config["map_size"] == 16 and config["train_config"] == "MAZE_LR":
            model_file = f"hyper/AZTrain_env=16x16_MAZE_LR_evalpol=visit_iterations=150_budget=64_df=0.95_lr=0.003_nstepslr=2_c=0.2_seed={model_seed}/checkpoint.pth"

        for budget in budgets:
            eval_results = []  # Store results across evaluation seeds for a given training seed

            for seed in range(num_eval_seeds):
                config_copy = dict(hparams)
                config_copy["model_file"] = model_file
                config_copy["planning_budget"] = budget

                #print(f"Running evaluation for planning_budget={budget}")

                agent, train_env, tree_evaluation_policy, observation_embedding, planning_budget = agent_from_config(config_copy)
                test_env = gym.make(**config_copy["test_env"])
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
                    azdetection=(config_copy["agent_type"] == "mini-trees" or config_copy["agent_type"] == "mega-tree"),
                    unroll_budget=config_copy["unroll_budget"],
                )

                episode_returns, discounted_returns, time_steps, _ = calc_metrics(
                    results, agent.discount_factor, test_env.action_space.n
                )

                eval_results.append([
                    discounted_returns.mean().item(),
                    episode_returns.mean().item(),
                    time_steps.mean().item()
                ])

            # Compute mean across evaluation seeds for this training seed
            eval_results = np.array(eval_results)
            train_seed_mean = eval_results.mean(axis=0)  # Mean of evaluation seeds
            results_data.append([budget, model_seed] + list(train_seed_mean))

    if use_wandb:
        run.finish()

    # Convert results into DataFrame
    df = pd.DataFrame(results_data, columns=["Budget", "Training Seed", "Discounted Return", "Return", "Episode Length"])

    # Compute final mean and standard deviation across training seeds
    df_grouped = df.groupby("Budget").agg(["mean", "std"])
    df_grouped.columns = [f"{col[0]} {col[1]}" for col in df_grouped.columns]  # Flatten MultiIndex
    df_grouped.reset_index(inplace=True)  # Restore "Budget" as a column

    # Print to debug column names
    print("Column names after grouping:", df_grouped.columns)

    # Compute standard error across training seeds
    num_train_seeds = len(df["Training Seed"].unique())
    for metric in ["Discounted Return", "Return", "Episode Length"]:
        df_grouped[f"{metric} SE"] = df_grouped[f"{metric} std"] / np.sqrt(num_train_seeds)
    
    # Drop the "Training Seed mean" column and the "Training Seed std" column
    df_grouped.drop(columns=["Training Seed mean", "Training Seed std"], inplace=True)
    if save:
        if final:
            # Check if the folder final exists
            if not os.path.exists("final"):
                os.makedirs("final")
            # Check if the folder map_size exists
            if not os.path.exists(f"final/{config_copy['map_size']}x{config_copy['map_size']}"):
                os.makedirs(f"final/{config_copy['map_size']}x{config_copy['map_size']}")

            # Save results
            df_grouped.to_csv(f"final/{config_copy['map_size']}x{config_copy['map_size']}/{run_name}.csv", index=False)

        else:
            # If directory does not exist, create it
            if not os.path.exists(f"{config_copy['map_size']}x{config_copy['map_size']}"):
                os.makedirs(f"{config_copy['map_size']}x{config_copy['map_size']}")

            # Save results
            df_grouped.to_csv(f"{config_copy['map_size']}x{config_copy['map_size']}/{run_name}.csv", index=False)

    # Print final averages with standard errors
    print(f"Final results for {run_name}")
    for budget in budgets:
        row = df_grouped[df_grouped["Budget"] == budget]
        print(f"Planning Budget: {budget}")
        print(f"Avg Discounted Return: {row['Discounted Return mean'].values[0]:.3f} ± {row['Discounted Return SE'].values[0]:.3f}")
        print(f"Avg Return: {row['Return mean'].values[0]:.3f} ± {row['Return SE'].values[0]:.3f}")
        print(f"Avg Episode Length: {row['Episode Length mean'].values[0]:.3f} ± {row['Episode Length SE'].values[0]:.3f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="AlphaZero Evaluation Configuration")

    map_size = 8

    TRAIN_CONFIG = "NO_HOLES" # NO_HOLES, MAZE_RL, MAZE_LR

    TEST_CONFIG = "SLALOM"

    parser.add_argument("--ENV", type=str, default="LUNARLANDER", help="Environment name")
    
    parser.add_argument("--map_size", type=int, default= map_size, help="Map size")
    parser.add_argument("--test_config", type=str, default= TEST_CONFIG, help="Config desc name")
    parser.add_argument("--train_config", type=str, default= TRAIN_CONFIG, help="Config desc name")

    # Run configurations
    parser.add_argument("--wandb_logs", type=bool, default= False, help="Enable wandb logging")
    parser.add_argument("--workers", type=int, default= 1, help="Number of workers")
    parser.add_argument("--runs", type=int, default= 1, help="Number of runs")

    # Basic search parameters
    parser.add_argument("--tree_evaluation_policy", type= str, default="visit", help="Tree evaluation policy")
    parser.add_argument("--selection_policy", type=str, default="PUCT", help="Selection policy")
    parser.add_argument("--puct_c", type=float, default= 1, help="PUCT parameter")

    # Only relevant for single run evaluation
    parser.add_argument("--planning_budget", type=int, default = 256, help="Planning budget")

    # Search algorithm
    parser.add_argument("--agent_type", type=str, default= "azmcts", help="Agent type")

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
    parser.add_argument("--value_search", type=bool, default=False, help="Enable value search")
    parser.add_argument("--predictor", type=str, default="current_value", help="Predictor to use for detection")
    parser.add_argument("--update_estimator", type=bool, default=True, help="Update the estimator")

    # Test environment
    parser.add_argument("--test_env_is_slippery", type=bool, default= False, help="Slippery environment")
    parser.add_argument("--test_env_hole_reward", type=int, default=0, help="Hole reward")
    parser.add_argument("--test_env_terminate_on_hole", type=bool, default= False, help="Terminate on hole")
    parser.add_argument("--deviation_type", type=str, default= "bump", help="Deviation type")

    parser.add_argument("--num_asteroids", type=int, default=0, help="Number of asteroids")

    # Model file
    parser.add_argument("--model_file", type=str, default= "", help="Model file")

    parser.add_argument( "--train_seeds", type=int, default=10, help="The number of random seeds to use for training.")
    parser.add_argument("--eval_seeds", type=int, default=1, help="The number of random seeds to use for evaluation.")

    # Rendering
    parser.add_argument("--render", type=bool, default=True, help="Render the environment")

    parser.add_argument("--run_full_eval", type=bool, default= False, help="Run type")

    parser.add_argument("--hpc", type=bool, default=False, help="HPC flag")

    parser.add_argument("--value_estimate", type=str, default="nn", help="Value estimate method")
    parser.add_argument("--visualize_trees", type=bool, default=False, help="Visualize trees")

    parser.add_argument("--var_penalty", type=float, default=1, help="Variance penalty")
    parser.add_argument("--value_penalty", type=float, default=0 , help="Value penalty")

    parser.add_argument("--final", type=bool, default=False)

    parser.add_argument("--save", type=bool, default=False)

    parser.add_argument("--bad_training", type=bool, default=False)

    parser.add_argument("--policy_det_rule", type=bool, default= False, help="Policy detection rule")

    parser.add_argument("--reuse_tree", type=bool, default=True, help="Update the estimator")

    parser.add_argument("--plot_tree_densities", type=bool, default=False, help="Plot tree densities")

    # Parse arguments
    args = parser.parse_args()

    single_seed = 3 # Only for single run

    ENV = args.ENV

    if ENV == "FROZENLAKE":
        args.test_env_id = f"CustomFrozenLakeNoHoles{args.map_size}x{args.map_size}-v1"
        args.test_env_desc = f"{args.map_size}x{args.map_size}_{args.test_config}"
        if args.map_size == 8 and args.train_config == "NO_HOLES":
            args.model_file = f"hyper/AZTrain_env=CustomFrozenLakeNoHoles8x8-v1_evalpol=visit_iterations=50_budget=64_df=0.95_lr=0.001_nstepslr=2_seed={single_seed}/checkpoint.pth"
        elif args.map_size == 16 and args.train_config == "NO_HOLES":
            args.model_file = f"hyper/AZTrain_env=CustomFrozenLakeNoHoles16x16-v1_evalpol=visit_iterations=60_budget=128_df=0.95_lr=0.003_nstepslr=2_seed={single_seed}/checkpoint.pth"
        elif args.map_size == 8 and args.train_config == "MAZE_RL":
            if args.bad_training:
                args.model_file = f"hyper/AZTrain_env=8x8_MAZE_RL_evalpol=visit_iterations=20_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={single_seed}/checkpoint.pth"
            else:
                args.model_file = f"hyper/AZTrain_env=8x8_MAZE_RL_evalpol=visit_iterations=150_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={single_seed}/checkpoint.pth"
        elif args.map_size == 8 and args.train_config == "MAZE_LR":
            if args.bad_training:
                args.model_file = f"hyper/AZTrain_env=8x8_MAZE_LR_evalpol=visit_iterations=10_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={single_seed}/checkpoint.pth"
            else:
                args.model_file = f"hyper/AZTrain_env=8x8_MAZE_LR_evalpol=visit_iterations=100_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={single_seed}/checkpoint.pth"
        elif args.map_size == 16 and args.train_config == "MAZE_LR":
            args.model_file = f"hyper/AZTrain_env=16x16_MAZE_LR_evalpol=visit_iterations=150_budget=64_df=0.95_lr=0.003_nstepslr=2_c=0.2_seed={single_seed}/checkpoint.pth"

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

    elif ENV == "LUNARLANDER":

        challenge = env_challenges["CustomLunarLander"]

        observation_embedding = "lunarlander"

        test_env_dict = {
            "id": "CustomLunarLander",
            "num_asteroids": args.num_asteroids,
        }

        args.model_file = "hyper/AZTrain_env=LunarLander_ast=0_evalpol=visit_iterations=1000_budget=64_df=0.99_lr=0.001_nstepslr=1_c=1_seed=None/checkpoint.pth"

        map_name = f"LunarLander-ast={args.num_asteroids}"

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
        "policy_det_rule": args.policy_det_rule,
        "reuse_tree": args.reuse_tree,
        "plot_tree_densities": args.plot_tree_densities,
    }

    run_config = {**base_parameters, **challenge, **config_modifications}

    # Execute the evaluation

    if args.run_full_eval:
        eval_budget_sweep(config=run_config, budgets= [64],  num_train_seeds=args.train_seeds, num_eval_seeds=args.eval_seeds, final = args.final, save=args.save)
    else:
        eval_from_config(config=run_config)
