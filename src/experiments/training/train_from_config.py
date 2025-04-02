import sys
import os

# ---------------------------------------------------------
# 1. Move the offline-flag check and environment variable
#    setting BEFORE importing wandb.
# ---------------------------------------------------------
if "--offline" in sys.argv:
    os.environ["WANDB_MODE"] = "offline"

sys.path.append("src/")
import datetime
import multiprocessing
import numpy as np
import gymnasium as gym
import argparse
from torch.utils.tensorboard.writer import SummaryWriter
import torch as th

from torchrl.data import (
    LazyTensorStorage,
    TensorDictReplayBuffer,
)

# ---------------------------------------------------------
# Now import wandb AFTER setting WANDB_MODE
# ---------------------------------------------------------
import wandb

import experiments.parameters as parameters
from environments.observation_embeddings import ObservationEmbedding, embedding_dict
from environments.minigrid.mini_grid import ObstaclesGridEnv
from environments.minigrid.utilities.wrappers import SparseActionsWrapper, gym_wrapper
from az.alphazero import AlphaZeroController
from az.azmcts import AlphaZeroMCTS
from azdetection.octopus import Octopus
from az.model import (
    AlphaZeroModel,
    activation_function_dict,
    norm_dict,
    models_dict,
)
from policies.tree_policies import tree_eval_dict
from policies.selection_distributions import selection_dict_fn
from policies.value_transforms import value_transform_dict

from environments.register import register_all

from experiments.parameters import base_parameters, env_challenges, fz_env_descriptions


def train_from_config(
    project_name="AlphaZeroTraining",
    entity=None,
    job_name=None,
    config=None,
    performance=True,
    tags=None,
    seed=None,
    offline=False,
):
    if tags is None:
        tags = []
    tags.append("training")

    if performance:
        tags.append("performance")

    # Define run_name before initializing wandb
    if config['ENV'] == "FROZENLAKE":
        run_name = f"AZTrain_env={config['name_config']}_evalpol={config['tree_evaluation_policy']}_iterations={config['iterations']}_budget={config['planning_budget']}_df={config['discount_factor']}_lr={config['learning_rate']}_nstepslr={config['n_steps_learning']}_c={config['puct_c']}_seed={seed}"
    elif config['ENV'] == "LUNARLANDER":
        run_name = f"{config['name_config']}_budget={config['planning_budget']}_c={config['puct_c']}_df={config['discount_factor']}_lr={config['learning_rate']}_n={config['n_steps_learning']}_vw={config['value_loss_weight']}_pw={config['policy_loss_weight']}_eperit={config['episodes_per_iteration']}_bufmul={config['replay_buffer_multiplier']}_norm={config['norm_layer']}_seed={seed}"

    # -----------------------------------------------------
    # 2. If offline, do NOT pass `entity=...` and pass `mode="offline"`.
    # -----------------------------------------------------
    if offline:
        # Make sure environment variable is set (just in case)
        os.environ["WANDB_MODE"] = "offline"
        # Also set mode="offline" in wandb.init
        run = wandb.init(
            project=project_name,
            name=run_name,  # Explicitly set the run name
            config=config,
            tags=tags,
            mode="offline",  # explicitly offline
        )
    else:
        # Normal (online) initialization
        settings = wandb.Settings(job_name=job_name)
        run = wandb.init(
            project=project_name,
            name=run_name,  # Explicitly set the run name
            entity=entity,    # only pass entity if online
            settings=settings,
            config=config,
            tags=tags,
        )

    assert run is not None
    hparams = wandb.config
    print(hparams)

    register_all()

    env = gym.make(**hparams["env_params"])
    if isinstance(env, ObstaclesGridEnv):
        env = gym_wrapper(env)

    print(env.observation_space)

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

    if "selection_value_transform" not in hparams or hparams["selection_value_transform"] is None:
        hparams["selection_value_transform"] = "identity"

    selection_policy = selection_dict_fn(
        hparams["puct_c"],
        tree_evaluation_policy,
        discount_factor,
        value_transform_dict[hparams["selection_value_transform"]],
    )[hparams["selection_policy"]]

    if "root_selection_policy" not in hparams or hparams["root_selection_policy"] is None:
        hparams["root_selection_policy"] = hparams["selection_policy"]

    root_selection_policy = selection_dict_fn(
        hparams["puct_c"],
        tree_evaluation_policy,
        discount_factor,
        value_transform_dict[hparams["selection_value_transform"]],
    )[hparams["root_selection_policy"]]

    if "observation_embedding" not in hparams:
        hparams["observation_embedding"] = "default"

    observation_embedding: ObservationEmbedding = embedding_dict[hparams["observation_embedding"]](
        env.observation_space,
        hparams["ncols"] if "ncols" in hparams else None,
    )

    print(type(observation_embedding))

    model: AlphaZeroModel = models_dict[hparams["model_type"]](
        env,
        observation_embedding=observation_embedding,
        hidden_dim=hparams["hidden_dim"],
        nlayers=hparams["layers"],
        activation_fn=activation_function_dict[hparams["activation_fn"]],
        norm_layer=norm_dict[hparams["norm_layer"]],
    )

    if "dir_epsilon" not in hparams:
        hparams["dir_epsilon"] = 0.0
        hparams["dir_alpha"] = None

    dir_epsilon = hparams["dir_epsilon"]
    dir_alpha = hparams["dir_alpha"]

    if hparams["agent_type"] == "azmcts":
        agent = AlphaZeroMCTS(
            root_selection_policy=root_selection_policy,
            selection_policy=selection_policy,
            model=model,
            dir_epsilon=dir_epsilon,
            dir_alpha=dir_alpha,
            discount_factor=discount_factor,
        )
    elif hparams["agent_type"] == "octopus":
        agent = Octopus(
            model=model,
            selection_policy=selection_policy,
            threshold=0.05,
            discount_factor=discount_factor,
            dir_epsilon=dir_epsilon,
            dir_alpha=dir_alpha,
            root_selection_policy=root_selection_policy,
            predictor="current_value",
            value_estimate="nn",
            var_penalty=1.0,
            value_penalty=1.0,
            update_estimator=True,
            policy_det_rule=False,
        )

    optimizer = th.optim.Adam(
        model.parameters(),
        lr=hparams["learning_rate"],
        weight_decay=hparams["regularization_weight"],
    )

    if "workers" not in hparams or hparams["workers"] is None:
        hparams["workers"] = multiprocessing.cpu_count()
    workers = hparams["workers"]

    if "episodes_per_iteration" not in hparams or hparams["episodes_per_iteration"] is None:
        hparams["episodes_per_iteration"] = workers
    episodes_per_iteration = hparams["episodes_per_iteration"]

    replay_buffer_size = hparams["replay_buffer_multiplier"] * episodes_per_iteration
    sample_batch_size = replay_buffer_size // hparams["sample_batch_ratio"]

    replay_buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(replay_buffer_size))

    log_dir = f"./tensorboard_logs/hyper/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)
    run_dir = f"./hyper/{run_name}"

    controller = AlphaZeroController(
        env,
        agent,
        optimizer,
        replay_buffer=replay_buffer,
        max_episode_length=hparams["max_episode_length"],
        planning_budget=hparams["planning_budget"],
        training_epochs=hparams["training_epochs"],
        value_loss_weight=hparams["value_loss_weight"],
        policy_loss_weight=hparams["policy_loss_weight"],
        reg_loss_weight=hparams["reg_loss_weight"],
        run_dir=run_dir,
        episodes_per_iteration=episodes_per_iteration,
        tree_evaluation_policy=tree_evaluation_policy,
        self_play_workers=workers,
        scheduler=th.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hparams["lr_gamma"], verbose=True),
        discount_factor=discount_factor,
        n_steps_learning=hparams["n_steps_learning"],
        checkpoint_interval=-1 if performance else 10,
        use_visit_count=bool(hparams["use_visit_count"]),
        writer=writer,
        save_plots=not performance,
        batch_size=sample_batch_size,
    )
    iterations = hparams["iterations"]
    temp_schedule = [None] * iterations
    metrics = controller.iterate(temp_schedule=temp_schedule, seed=seed)

    env.close()

    # If you are offline, skip code-snapshot or do it if you want local artifact
    if not offline:
        run.log_code(root="./src")

    run.finish()
    return metrics


def sweep_agent():
    train_from_config(performance=True)


def run_single(seed=None):
    return train_from_config(config=run_config, performance=False, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero Training with a specific seed.")
    parser.add_argument("--ENV", type=str, default="LUNARLANDER", help="The environment to train on.")
    parser.add_argument("--agent_type", type=str, default="azmcts", help="The type of agent to train.")
    parser.add_argument("--workers", type=int, default=6, help="Number of workers")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--max_episode_length", type=int, default=1000, help="Max episode length")
    parser.add_argument("--tree_evaluation_policy", type=str, default="visit", help="Tree evaluation policy")
    parser.add_argument("--selection_policy", type=str, default="PUCT", help="Selection policy")
    parser.add_argument("--planning_budget", type=int, default=128, help="Planning budget")
    parser.add_argument("--puct_c", type=float, default=1, help="PUCT constant")
    parser.add_argument("--n_steps_learning", type=int, default=1, help="Number of steps for learning")
    parser.add_argument("--discount_factor", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--value_loss_weight", type=float, default=1, help="Value loss weight")
    parser.add_argument("--policy_loss_weight", type=float, default=100, help="Policy loss weight")
    parser.add_argument("--reg_loss_weight", type=float, default=0.0, help="Regularization loss weight")
    parser.add_argument("--replay_buffer_multiplier", type=int, default=15, help="Replay buffer multiplier")
    parser.add_argument("--episodes_per_iteration", type=int, default=15, help="Episodes per iteration")
    parser.add_argument("--norm_layer", type=str, default="batch_norm", help="Normalization layer")
    parser.add_argument("--scale_reward", type=bool, default=False, help="Scale reward")
    parser.add_argument("--map_size", type=int, default=8, help="The size of the map.")
    parser.add_argument("--train_env_desc", type=str, default="8x8_MAZE_RL", help="Environment description.")
    parser.add_argument("--train_slippery", type=bool, default=False, help="Whether the environment is slippery.")
    parser.add_argument("--train_hole_reward", type=float, default=0.0, help="Reward for falling into a hole.")
    parser.add_argument("--train_terminate_on_hole", type=bool, default=False, help="Terminate if hole is encountered?")
    parser.add_argument("--train_deviation_type", type=str, default="bump", help="The type of deviation to use.")
    parser.add_argument("--num_asteroids", type=int, default=0, help="Number of asteroids")
    parser.add_argument("--train_seed", type=int, default=0, help="The random seed to use for training.")

    # ---------------------------------------------------------
    # 3. Set default to False so you can pass --offline to enable
    # ---------------------------------------------------------
    parser.add_argument("--offline", action="store_true", help="Run in offline mode (no W&B sync).")

    args = parser.parse_args()

    ENV = args.ENV

    if ENV == "FROZENLAKE":
        challenge = env_challenges[f"CustomFrozenLakeNoHoles{args.map_size}x{args.map_size}-v1"]
        challenge["env_params"]["desc"] = fz_env_descriptions[args.train_env_desc]
        challenge["env_params"]["is_slippery"] = args.train_slippery
        challenge["env_params"]["hole_reward"] = args.train_hole_reward
        challenge["env_params"]["terminate_on_hole"] = args.train_terminate_on_hole
        challenge["env_params"]["deviation_type"] = args.train_deviation_type
        name_config = args.train_env_desc
        observation_embedding = "coordinate"
    elif ENV == "LUNARLANDER":
        challenge = env_challenges["CustomLunarLander"]
        challenge["env_params"]["num_asteroids"] = args.num_asteroids
        challenge["env_params"]["scale_reward"] = args.scale_reward
        name_config = f"LunarLander_ast={args.num_asteroids}"
        observation_embedding = "lunarlander"

    config_modifications = {
        "ENV": ENV,
        "workers": args.workers,
        "tree_evaluation_policy": args.tree_evaluation_policy,
        "selection_policy": args.selection_policy,
        "planning_budget": args.planning_budget,
        "iterations": args.iterations,
        "observation_embedding": observation_embedding,
        "n_steps_learning": args.n_steps_learning,
        "name_config": name_config,
        "puct_c": args.puct_c,
        "agent_type": args.agent_type,
        "discount_factor": args.discount_factor,
        "learning_rate": args.learning_rate,
        "value_loss_weight": args.value_loss_weight,
        "policy_loss_weight": args.policy_loss_weight,
        "reg_loss_weight": args.reg_loss_weight,
        "replay_buffer_multiplier": args.replay_buffer_multiplier,
        "episodes_per_iteration": args.episodes_per_iteration,
        "max_episode_length": args.max_episode_length,
        "offline": args.offline,
    }

    run_config = {**base_parameters, **challenge, **config_modifications}

    train_from_config(
        config=run_config,
        performance=False,
        seed=args.train_seed,
        offline=args.offline
    )