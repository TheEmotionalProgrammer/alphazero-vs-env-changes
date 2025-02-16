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
        "optimal_value": 1.0 * 0.9 ** 6,
        "worst_value": 0.0,
        "discount_factor": 0.9,
        "ncols": 4,
    },

    "CustomFrozenLakeNoHoles8x8-v1": {
        "env_description": "CustomFrozenLakeNoHoles8x8-v1",
        "max_episode_length": 30,
        "env_params": dict(id="CustomFrozenLakeNoHoles8x8-v1", max_episode_steps=1000000000),
        "optimal_value": 1.0 * 0.95 ** 14,
        "worst_value": 0.0,
        "discount_factor": 0.95,
        "eval_param": 10.0,
        "ncols": 8,
    },

    "CustomFrozenLakeNoHoles16x16-v1": {
        "env_description": "CustomFrozenLakeNoHoles16x16-v1",
        "max_episode_length": 50,
        "env_params": dict(id="CustomFrozenLakeNoHoles16x16-v1", max_episode_steps=1000000000),
        "optimal_value": 1.0 * 0.95 ** 30,
        "worst_value": 0.0,
        "discount_factor": 0.95,
        "eval_param": 10.0,
        "ncols": 16,
        "learning_rate": 3e-3,
    },

    "CustomFrozenLakeNoHoles20x20-v1": {
        "env_description": "CustomFrozenLakeNoHoles20x20-v1",
        "max_episode_length": 400,
        "env_params": dict(id="CustomFrozenLakeNoHoles20x20-v1", max_episode_steps=1000000000),
        "optimal_value": 1.0 * 0.95 ** 38,
        "worst_value": 0.0,
        "discount_factor": 0.95,
        "eval_param": 10.0,
        "ncols": 20,

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
        "HHHHHHHF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFHHH",
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

    "8x8_DEFAULT": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],

    "16x16_DEFAULT": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFHHFFFFFFFFFF",
        "FFFFHHFFFFFFFFFF",
        "FFFFFFFFHHFFFFFF",
        "FFFFFFFFHHFFFFFF",
        "FFFFHHFFFFFFFFFF",
        "FFFFHHFFFFFFFFFF",
        "FFFFFFFFFFFFHHFF",
        "FFHHFFFFFFFFHHFF",
        "FFHFFFFFHHFFHHFF",
        "FFHFFFFFHHFFHHFF",
        "FFHFFFFFFFFFFFFF",
        "FFFFFHHFFFFFFFFF",
        "FFFFFHHFFFFFFFFG"
    ],

    "8x8_NARROW": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HHFHHHHH",
        "HHFHHHHH",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "16x16_NARROW": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHHHFFHHHHHHHHHH",
        "HHHHFFHHHHHHHHHH",
        "HHHHFFHHHHHHHHHH",
        "HHHHFFHHHHHHHHHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],

    "8x8_NARROW_XTREME": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HHFHHHHH",
        "HHFHHHHH",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFHF",
        "FFFFFFHG"
    ],

    "16x16_NARROW_XTREME": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHHHFFHHHHHHHHHH",
        "HHHHFFHHHHHHHHHH",
        "HHHHFFHHHHHHHHHH",
        "HHHHFFHHHHHHHHHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFHHFFFF",
        "FFFFFFFFFFHHFFFF",
        "FFFFFFFFFFHHFFFF",
        "FFFFFFFFFFHHFFFG"
    ],

    "8x8_DEAD_END": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFHHHHHH",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "16x16_DEAD_END": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFHHHHHHHHHHHH",
        "FFFFHHHHHHHHHHHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
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

    "16x16_NO_OBSTACLES": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],

    "16x16_BLOCK": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFHHHHHHHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],

    "16x16_IMPOSSIBLE": [ # for debugging purposes
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHHHHHHHHHHHHHHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],

    # For detection experiments

    "16x16_D2": [
        "SFFHFFFFFFFFFFFF",
        "FFFHFFFFFFFFFFFF",
        "FFFHFFFFFFFFFFFF",
        "HHHHFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],
    
    "16x16_D5": [
        "SFFFFFHFFFFFFFFF",
        "FFFFFFHFFFFFFFFF",
        "FFFFFFHFFFFFFFFF",
        "FFFFFFHFFFFFFFFF",
        "FFFFFFHFFFFFFFFF",
        "FFFFFFHFFFFFFFFF",
        "HHHHHHHFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],

    "16x16_D8": [
        "SFFFFFFFFHFFFFFF",
        "FFFFFFFFFHFFFFFF",
        "FFFFFFFFFHFFFFFF",
        "FFFFFFFFFHFFFFFF",
        "FFFFFFFFFHFFFFFF",
        "FFFFFFFFFHFFFFFF",
        "FFFFFFFFFHFFFFFF",
        "FFFFFFFFFHFFFFFF",
        "FFFFFFFFFHFFFFFF",
        "HHHHHHHHHHFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],

    "16x16_D11": [
        "SFFFFFFFFFFFHFFF",
        "FFFFFFFFFFFFHFFF",
        "FFFFFFFFFFFFHFFF",
        "FFFFFFFFFFFFHFFF",
        "FFFFFFFFFFFFHFFF",
        "FFFFFFFFFFFFHFFF",
        "FFFFFFFFFFFFHFFF",
        "FFFFFFFFFFFFHFFF",
        "FFFFFFFFFFFFHFFF",
        "FFFFFFFFFFFFHFFF",
        "FFFFFFFFFFFFHFFF",
        "FFFFFFFFFFFFHFFF",
        "FFFFFFFFFFFFHFFF",
        "HHHHHHHHHHHHHFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],

    "16x16_D14": [
        "SFFFFFFFFFFFFFFH",
        "FFFFFFFFFFFFFFFH",
        "FFFFFFFFFFFFFFFH",
        "FFFFFFFFFFFFFFFH",
        "FFFFFFFFFFFFFFFH",
        "FFFFFFFFFFFFFFFH",
        "FFFFFFFFFFFFFFFH",
        "FFFFFFFFFFFFFFFH",
        "FFFFFFFFFFFFFFFH",
        "FFFFFFFFFFFFFFFH",
        "FFFFFFFFFFFFFFFH",
        "FFFFFFFFFFFFFFFH",
        "FFFFFFFFFFFFFFFH",
        "FFFFFFFFFFFFFFFH",
        "FFFFFFFFFFFFFFFH",
        "HHHHHHHHHHHHHHHG"

    ],

}
