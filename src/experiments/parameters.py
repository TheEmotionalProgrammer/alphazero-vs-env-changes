base_parameters = {
    "model_type": "seperated",
    "observation_embedding": "default",
    "activation_fn": "relu",
    "norm_layer": "batch_norm",
    "dir_epsilon": 0.4,
    "dir_alpha": 2.5,
    "selection_policy": "PUCT",
    "puct_c": 1.0,
    "selection_value_transform": "identity",
    "use_visit_count": False,
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
    "max_episode_length": 1000,
    "episodes_per_iteration": 6,
    "eval_temp": 0,
}

env_challenges = {

    "CustomLunarLander": {
        "env_description": "CustomLunarLander",
        "env_params": dict(id="CustomLunarLander", max_episode_steps=1000000000),
        "optimal_value": 200.0,
        "worst_value": -200.0,
    },

    "ParkingEnv": {
        "env_description": "ParkingEnv",
        "env_params": dict(id="ParkingEnv", max_episode_steps=1000000000),
        "optimal_value": 0.0,
        "worst_value": -1.0,
    },

    "CustomFrozenLakeNoHoles4x4-v1": {
        "env_description": "CustomFrozenLakeNoHoles4x4-v1",
        "max_episode_length": 40,
        "iterations": 30,
        "env_params": dict(id="CustomFrozenLakeNoHoles4x4-v1", max_episode_steps=1000000000),
        "optimal_value": 1.0 * 0.9 ** 6,
        "worst_value": 0.0,
        "discount_factor": 0.9,
        "ncols": 4,
    },

    "CustomFrozenLakeNoHoles8x8-v1": {
        "env_description": "CustomFrozenLakeNoHoles8x8-v1",
        "max_episode_length": 100,
        "env_params": dict(id="CustomFrozenLakeNoHoles8x8-v1", max_episode_steps=1000000000),
        "optimal_value": 1.0 * 0.95 ** 14,
        "worst_value": 0.0,
        "discount_factor": 0.95,
        "ncols": 8,
        "learning_rate": 1e-3,
    },

    "CustomFrozenLakeNoHoles16x16-v1": {
        "env_description": "CustomFrozenLakeNoHoles16x16-v1",
        "max_episode_length": 100,
        "env_params": dict(id="CustomFrozenLakeNoHoles16x16-v1", max_episode_steps=1000000000),
        "optimal_value": 1.0 * 0.95 ** 30,
        "worst_value": 0.0,
        "discount_factor": 0.95,
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

ll_env_descriptions = {
    "NO_ASTEROIDS": {
        "num_asteroids": 0,
        "ast_positions": None,
        "ast_shapes": None,
        "ast_sizes": None,
    },

    "SINGLE_CENTRAL_PENTAGON" : {
        "num_asteroids": 1,
        "ast_positions": [(0.5, 0.6)],
        "ast_shapes": ["pentagon", "pentagon"],
        "ast_sizes": [15],
    },

    "TWO_LATERAL_PENTAGONS" : {
        "num_asteroids": 2,
        "ast_positions": [(0.4, 0.6), (0.6, 0.6)],
        "ast_shapes": ["pentagon", "pentagon"],
        "ast_sizes": [15, 15],
    },

    "RIGHT_PENTAGON" : {
        "num_asteroids": 1,
        "ast_positions": [(0.6, 0.6)],
        "ast_shapes": ["pentagon"],
        "ast_sizes": [15],
    },
}

fz_env_descriptions = {
    
    "8x8_NO_OBSTACLES": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "8x8_SLALOM": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HHHHHHHF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFHH",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "16x16_SLALOM": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHHHHHHHHHHHHHFF",
        "HHHHHHHHHHHHHHFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFHHHH",
        "FFFFFFFFFFFFHHHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
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
        "SFFFFFHH",
        "FFFFFFHH",
        "HHFHHHHH",
        "HHFHHHHH",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "16x16_NARROW": [
        "SFFFFFFFFFFFHHHH",
        "FFFFFFFFFFFFHHHH",
        "FFFFFFFFFFFFHHHH",
        "FFFFFFFFFFFFHHHH",
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

    "8x8_MAZE_RL": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HHHHHFFH",
        "FFFFFFFF",
        "FFFFFFFF",
        "HFFHHHHH",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "8x8_MAZE_LR": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HFFHHHHH",
        "FFFFFFFF",
        "FFFFFFFF",
        "HHHHHFFH",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "8x8_MAZE_LL": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HFFHHHHH",
        "FFFFFFFF",
        "FFFFFFFF",
        "HFFHHHHH",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "8x8_MAZE_RR": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HHHHHFFH",
        "FFFFFFFF",
        "FFFFFFFF",
        "HHHHHFFH",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "8x8_MAZE_RC": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HHHHHFFH",
        "FFFFFFFF",
        "FFFFFFFF",
        "HHHFFHHH",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "8x8_MAZE_LC": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HFFHHHHH",
        "FFFFFFFF",
        "FFFFFFFF",
        "HHHFFHHH",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "16x16_MAZE_RL": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHHHHHHHHHFFFFHH",
        "HHHHHHHHHHFFFFHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHFFFFHHHHHHHHHH",
        "HHFFFFHHHHHHHHHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],

    "16x16_MAZE_LR": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHFFFFHHHHHHHHHH",
        "HHFFFFHHHHHHHHHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHHHHHHHHHFFFFHH",
        "HHHHHHHHHHFFFFHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],

    "16x16_MAZE_LR_OBS": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHFFFFHHHHHHHHHH",
        "HHFFFFHHHHHHHHHH",
        "FFFFFFHFHFHFFFFF",
        "FFFFFFHFFFHFFFFF",
        "FFFFFFFFHFFFFFFF",
        "FFFFFFHFHFHFFFFF",
        "HHHHHHHHHHHFFFHH",
        "HHHHHHHHHHHFFFHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],

    "16x16_MAZE_LL": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHFFFFHHHHHHHHHH",
        "HHFFFFHHHHHHHHHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHFFFFHHHHHHHHHH",
        "HHFFFFHHHHHHHHHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],

    "16x16_MAZE_RR": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHHHHHHHHHFFFFHH",
        "HHHHHHHHHHFFFFHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHHHHHHHHHFFFFHH",
        "HHHHHHHHHHFFFFHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
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
