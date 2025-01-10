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
}

env_challenges = {

    "CustomFrozenLakeNoHoles4x4-v1": {
        "env_description": "CustomFrozenLakeNoHoles4x4-v1",
        "max_episode_length": 100,
        "iterations": 30,
        "env_params": dict(id="CustomFrozenLakeNoHoles4x4-v1", max_episode_steps=1000000000),
        "optimal_value": 1.0 * 0.9 ** 5,
        "worst_value": 0.0,
        "discount_factor": 0.9,
        "ncols": 4,
    },

    "CustomFrozenLakeNoHoles8x8-v1": {
        "env_description": "CustomFrozenLakeNoHoles8x8-v1",
        "max_episode_length": 100,
        "env_params": dict(id="CustomFrozenLakeNoHoles8x8-v1", max_episode_steps=1000000000),
        "optimal_value": 1.0 * 0.95 ** 13,
        "worst_value": 0.0,
        "discount_factor": 0.95,
        "eval_param": 10.0,
        "ncols": 8,
    },

    "DefaultFrozenLake4x4-v1": {
        "env_description": "DefaultFrozenLake4x4-v1",
        "max_episode_length": 100,
        "iterations": 30,
        "env_params": dict(id="DefaultFrozenLake4x4-v1", max_episode_steps=1000000000),
        "optimal_value": 1.0 * 0.9 ** 5,
        "worst_value": -1,
        "discount_factor": 0.9,
        "ncols": 4,
    },

    "DefaultFrozenLake8x8-v1": {
        "env_description": "DefaultFrozenLake8x8-v1",
        "max_episode_length": 100,
        "env_params": dict(id="DefaultFrozenLake8x8-v1", max_episode_steps=1000000000),
        "optimal_value": 1.0 * 0.95 ** 13,
        "worst_value": 0,
        "discount_factor": 0.95,
        "ncols": 8,
    },
}

fz_env_descriptions = {
    
    "NO_OBSTACLES": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "DEFAULT": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],

    "MINI_SLALOM": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HHHHFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFHHH",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "SLALOM": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HHHHFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFHHHH",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "MINI_BLOCKS": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFHHFFFF",
        "FFFFFHHF",
        "HHFFFFFF",
        "FFFFFFFF",
        "FFFFFHHF",
        "FFFFFFFG"
    ],

    "BLOCKS": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFHHHFFF",
        "FFFFFFFF",
        "HHHFFFFF",
        "FFFFFFFF",
        "FFFFFHHH",
        "FFFFFFFG"
    ],

    "NARROW": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HHFHHHHH",
        "HHFHHHHH",
        "FFFFFFFF",
        "FFFFFFHF",
        "FFFFFFHF",
        "FFFFFFHG"
    ],

    "NARROW_SIMPLIFIED": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HHFHHHHH",
        "HHFHHHHH",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "DEAD_END": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFHHHHHH",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "INVERSE_DEAD_END": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "HHHHHHFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFG"
    ],
    
    "TRAP": [
        "SFFFFFFF",
        "FFFHFFFF",
        "HHHHFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "COLUMN": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFHHFFFF",
        "FFHHFFFF",
        "FFHHFFFF",
        "FFHHFFFF",
        "FFHHFFFF",
        "FFFFFFFG"
    ],

}
