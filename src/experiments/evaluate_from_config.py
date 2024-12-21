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
from az.azmcts import AlphaZeroMCTS
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

    # Initialize Weights & Biases
    settings = wandb.Settings(job_name=job_name)
    run = wandb.init(
        project=project_name, entity=entity, settings=settings, config=config, tags=tags
    )
    assert run is not None
    hparams = wandb.config

    agent, train_env, tree_evaluation_policy, observation_embedding, planning_budget = (
        agent_from_config(hparams)
    )

    if "workers" not in hparams or hparams["workers"] is None:
        hparams["workers"] = multiprocessing.cpu_count()
    workers = hparams["workers"]

    seeds = [None] * hparams["runs"]

    if isinstance(hparams["test_env"], str) and hparams["test_env"] == "train_env": # Use the training environment as the test environment

        test_env = copy.deepcopy(train_env)

    elif isinstance(hparams["test_env"], dict): # Use the given test environment

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
        "Evaluation/Returns": wandb.Histogram(np.array((episode_returns))),
        "Evaluation/Discounted_Returns": wandb.Histogram(
            np.array((discounted_returns))
        ),
        "Evaluation/Timesteps": wandb.Histogram(np.array((time_steps))),
        # "Evaluation/Entropies": wandb.Histogram(
        #     np.array(((th.sum(entropies, dim=-1) / time_steps)))
        # ),
        "Evaluation/Mean_Returns": episode_returns.mean().item(),
        "Evaluation/Mean_Discounted_Returns": discounted_returns.mean().item(),
        # "Evaluation/Mean_Entropy": (th.sum(entropies, dim=-1) / time_steps)
        # .mean()
        # .item(),
        "trajectories": trajectories,
    }
    run.log(data=eval_res)
    run.log_code(root="./src")
    # Finish the WandB run
    run.finish()


def eval_single():

    register_all()

    challenge = parameters.env_challenges[3] # Training environment

    config_modifications = {
        "workers": min(6, multiprocessing.cpu_count()),

        "tree_evaluation_policy": "visit",
        "selection_policy": "UCT",
        "runs": 1,
        "planning_budget": 128,
        "observation_embedding": "coordinate",
        "agent_type": "azmcts",
    
        "threshold": 0.03, # Only for azdetection, ignored otherwise
        "unroll_budget": 10, # Only for azdetection, ignored otherwise

        "eval_temp":0,
        "dir_epsilon": 0.0,
        "dir_alpha": None,

        "planning_style": "mini_trees",
        "predictor": "original_env", 

        #"test_env": None, # If None, the training environment is used

        #"test_env": "train_env", # If "train_env", the training environment is used

        "test_env": dict(    
            id = "DefaultFrozenLake8x8-v1",
            #id = "CustomFrozenLakeNoHoles8x8-v1",
            
            # desc = [ # MINI-SLALOM
            #     "SFFFFFFF",
            #     "FFFFFFFF",
            #     "FFFFFFFF",
            #     "HHHHFFFF",
            #     "FFFFFFFF",
            #     "FFFFFHHH",
            #     "FFFFFFFF",
            #     "FFFFFFFG",
            # ],
            # desc = [ # BLOCKS
            #     "SFFFFFFF",
            #     "FFFFFFFF",
            #     "FFHHHFFF",
            #     "FFFFFFFF",
            #     "HHHFFFFF",
            #     "FFFFFFFF",
            #     "FFFFFHHH",
            #     "FFFFFFFG",
            # ],
            # desc = [  # NARROW
            #     "SFFFFFFF",
            #     "FFFFFFFF",
            #     "HHFHHHHH",
            #     "HHFHHHHH",
            #     "FFFFFFFF",
            #     "FFFFFFHF",
            #     "FFFFFFHF",
            #     "FFFFFFHG",
            # ],
            # desc = [ # DEAD-END
            #     "SFFFFFFF",
            #     "FFFFFFFF",
            #     "FFFFFFFF",
            #     "FFFFFFFF",
            #     "FFFFFFFF",
            #     "FFHHHHHH",
            #     "FFFFFFFF",
            #     "FFFFFFFG",
            # ],
            # desc = [ # DEFAULT      
            #     "SFFFFFFF",
            #     "FFFFFFFF",
            #     "FFFHFFFF",
            #     "FFFFFHFF",
            #     "FFFHFFFF",
            #     "FHHFFFHF",
            #     "FHFFHFHF",
            #     "FFFHFFFG",
            # ],
            desc = [ # TRAP
                "SFFFFFFF",
                "FFFHFFFF",
                "HHHHFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFG",
            ],
            is_slippery=False,
            hole_reward=0,
            terminate_on_hole=False,
           
        ),

        "model_file": "/Users/isidorotamassia/THESIS/alphazero-vs-env-changes/runs/hyper/CustomFrozenLakeNoHoles8x8-v1_20241216-003012/checkpoint.pth",
    }

    run_config = {**parameters.base_parameters, **challenge, **config_modifications}

    return eval_from_config(config=run_config)


def custom_eval_sweep():
    challenge = parameters.env_challenges[3]
    config_modifications = {
        "workers": 6,
        "runs": 100,
        "agent_type": "distance",
        }
    series_configs = [
        {"tree_evaluation_policy": "visit", "selection_policy": "UCT"},
        {'tree_evaluation_policy': 'mvc', 'selection_policy': 'UCT'},
        {'tree_evaluation_policy': 'mvc', 'selection_policy': 'PolicyUCT'},
    ]
    # series_configs = [
    #     {"puct_c": x} for x in [1e-2, 1e0, 1e2]
    # ]

    run_config = {**parameters.base_parameters, **challenge, **config_modifications}

    budget_configs = [{"planning_budget": 2**i} for i in range(4, 8)]
    configs = [
        {**run_config, **variable_config, **series_config}
        for variable_config in budget_configs
        for series_config in series_configs
    ]
    print(f"Number of runs: {len(configs)}")

    time_name = time.strftime("%Y-%m-%d-%H-%M-%S")

    tags = ["eval_sweep", time_name]

    for config in tqdm(configs):
        eval_from_config(config=config, tags=tags)


if __name__ == "__main__":
    # sweep_id = wandb.sweep(sweep=coord_search, project="AlphaZero")

    # wandb.agent(sweep_id, function=sweep_agent)
    eval_single()
