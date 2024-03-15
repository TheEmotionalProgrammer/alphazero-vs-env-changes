def selection_to_expansion(selection_policy):
    """
    There is a connection between selection policy and expansion policy.
    puct -> fromprior
    policy_puct -> fromprior
    default -> default
    policy_uct -> default
    """
    if selection_policy in ["UCT", "PolicyUCT"]:
        return "default"
    else:
        return "fromprior"


selection = "PUCT"
base_parameters = {
    "model_type": "seperated",
    "observation_embedding": "default",
    "expansion_policy": selection_to_expansion(selection),
    "activation_fn": "relu",
    "norm_layer": "none",
    "dir_epsilon": 0.1,
    "dir_alpha": 0.5,
    "selection_policy": selection,
    # "root_seleciton_policy": selection,
    "puct_c": 1.0,
    "use_visit_count": 1,
    "regularization_weight": 0,
    "tree_evaluation_policy": "visit",
    "eval_param": 1.0,
    "tree_temperature": None,
    "hidden_dim": 64,
    "learning_rate": 3e-3,
    "sample_batch_ratio": 1,
    "n_steps_learning": 1,
    "training_epochs": 3,
    "planning_budget": 32,
    "layers": 5,
    "replay_buffer_multiplier": 10,
    "discount_factor": 1.0,
    "lr_gamma": 1.0,
    "iterations": 40,
    "policy_loss_weight": 0.5,
    "value_loss_weight": 0.5,
    "max_episode_length": 200,
    "episodes_per_iteration": 6,
}


env_challenges = [
    {
        "env_description": "CartPole-v1-300-30",
        "max_episode_length": 300,
        "iterations": 30,
        "env_params": dict(id="CartPole-v1", max_episode_steps=None),
        "observation_embedding": "default",
        "ncols": None,
    },
    {
        "env_description": "CliffWalking-v0-100-15",
        "max_episode_length": 100,
        "iterations": 20,
        "env_params": dict(id="CliffWalking-v0", max_episode_steps=None),
        "observation_embedding": "coordinate",
        "ncols": 12,
    },
    {
        "env_description": "FrozenLake-v1-4x4-150-20",
        "max_episode_length": 150,
        "iterations": 20,
        "env_params": dict(
            id="FrozenLake-v1",
            desc=None,
            map_name="4x4",
            is_slippery=False,
            max_episode_steps=None,
        ),
        "observation_embedding": "coordinate",
        "ncols": 4,
    },
    {
        "env_description": "FrozenLake-v1-8x8-150-20",
        "max_episode_length": 150,
        "iterations": 20,
        "env_params": dict(
            id="FrozenLake-v1",
            desc=None,
            map_name="8x8",
            is_slippery=False,
            max_episode_steps=None,
        ),
        "observation_embedding": "coordinate",
        "ncols": 8,
    },
]
