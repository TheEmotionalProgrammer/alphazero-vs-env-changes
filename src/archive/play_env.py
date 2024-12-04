import gymnasium as gym

def random_policy(env_name, max_steps=100):
    """
    Executes a random policy in the provided environment.

    Args:
        env_name: The name of the Gym environment to run.
        max_steps: Maximum number of steps to simulate.
    """
    env = gym.make(env_name, render_mode="human")  # Use "human" render mode for visualization
    obs, info = env.reset(seed=0)
    terminated = False
    truncated = False
    step = 0

    print(f"Running random policy on environment: {env_name}")
    while not terminated and not truncated and step < max_steps:
        action = env.action_space.sample()  # Generate a random action
        obs, rew, terminated, truncated, info = env.step(action)
        print(f"Step: {step}, O: {obs}, A: {action}, R: {rew}, T: {terminated}, Tr: {truncated}")
        step += 1
        env.render()  # Render the environment (optional)

    env.close()


if __name__ == "__main__":
    # List of supported environments
    available_envs = {
        "cw": "CliffWalking-v0",
        "fl": "FrozenLake-v1",
    }

    print("Available environments:")
    for env_key in available_envs.keys():
        print(f"- {env_key}")

    # Get user input for environment selection
    user_env_choice = input("Enter the environment you want to run (e.g., 'cw', 'fl'): ").strip().lower()

    if user_env_choice in available_envs:
        random_policy(available_envs[user_env_choice])
    else:
        print(f"Invalid choice. Please choose from {list(available_envs.keys())}.")