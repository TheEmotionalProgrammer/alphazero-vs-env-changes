base_parameters = {
    "model_type": "seperated",
    "observation_embedding": "default",
    "activation_fn": "relu",
    "norm_layer": "none",
    "dir_epsilon": 0.4,
    "dir_alpha": 2.5,
    "selection_policy": "PUCT",
    "puct_c": 1.0,
    "selection_value_transform": "identity",
    "use_visit_count": True,
    "regularization_weight": 1e-6,
    "tree_evaluation_policy": "visit",
    "eval_param": 10.0,
    "tree_temperature": None,
    "tree_value_transform": "identity",
    "hidden_dim": 64,
    "learning_rate": 1e-3,
    "sample_batch_ratio": 4,
    "n_steps_learning": 3,
    "training_epochs": 4,
    "planning_budget": 32,
    "layers": 2,
    "replay_buffer_multiplier": 15,
    "discount_factor": 0.99,
    "lr_gamma": 1.0,
    "iterations": 40,
    "policy_loss_weight": 0.3,
    "value_loss_weight": 0.7,
    "max_episode_length": 200,
    "episodes_per_iteration": 6,
    "eval_temp": 0,
    "test_env": "DefaultFrozenLake8x8-v1",
}

lake_config = {
    "max_episode_length": 100,
    "iterations": 30,
    "observation_embedding": "coordinate",
    "puct_c": 1.0,
    "eval_param": 10.0,
    "n_steps_learning": 1,
}

env_challenges = [
    {
        **lake_config,
        "env_description": "FrozenLake-v1-4x4",
        "ncols": 4,
        "env_params": dict(
            id="FrozenLake-v1",
            desc=[
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG"
            ],
            is_slippery=False,
            max_episode_steps=1000000000,
            hole_reward=-1.0,
        ),
        "discount_factor": (small_lake_discount_factor := 0.9),

        "optimal_value": 1.0 * small_lake_discount_factor ** 5,
        "worst_value": -1.0,
        "training_epochs": 2,
    },
    {
        "env_description": "MiniGrid-12x12-v0",
        "iterations": 20,
        "env_params": dict(id="MiniGrid-12x12-v0", max_steps=1000000000),
        "optimal_value": 1.0 * 0.99 ** 8,
        "worst_value": 0.0,
        "observation_embedding": "minigrid",
        "width": 12,
        "height": 12,
        "puct_c": 1.0,
        "eval_param": 1.0,
    },
    {
        "env_description": "CustomFrozenLakeNoHoles4x4-v1",
        "max_episode_length": 100,
        "iterations": 30,
        "env_params": dict(id="CustomFrozenLakeNoHoles4x4-v1", max_episode_steps=1000000000),
        "optimal_value": 1.0 * 0.9 ** 5,
        "worst_value": 0.0,
        "discount_factor": 0.9,
        "ncols": 4,
    },
    {
        "env_description": "CustomFrozenLakeNoHoles8x8-v1",
        "max_episode_length": 100,
        "iterations": 30,
        "env_params": dict(id="CustomFrozenLakeNoHoles8x8-v1", max_episode_steps=1000000000),
        "optimal_value": 1.0 * 0.95 ** 13,
        "worst_value": 0.0,
        "discount_factor": 0.95,
        "n_steps_learning": 2,
        "eval_param": 10.0,
        "ncols": 8,
    },
    {
        "env_description": "DefaultFrozenLake4x4-v1",
        "max_episode_length": 100,
        "iterations": 30,
        "env_params": dict(id="DefaultFrozenLake4x4-v1", max_episode_steps=1000000000),
        "optimal_value": 1.0 * 0.9 ** 5,
        "worst_value": -1,
        "discount_factor": 0.9,
        "ncols": 4,
    },
    {
        "env_description": "DefaultFrozenLake8x8-v1",
        "max_episode_length": 100,
        "iterations": 30,
        "env_params": dict(id="DefaultFrozenLake8x8-v1", max_episode_steps=1000000000),
        "optimal_value": 1.0 * 0.95 ** 13,
        "worst_value": 0,
        "discount_factor": 0.95,
        "n_steps_learning": 2,
        "ncols": 8,
    }

    
    
]
