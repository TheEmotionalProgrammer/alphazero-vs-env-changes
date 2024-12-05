import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from gymnasium.envs.registration import register


class CustomFrozenLakeEnv(FrozenLakeEnv):
    def __init__(
        self, desc=None, map_name="4x4", is_slippery=True, hole_penalty=-1
    ):
        super().__init__(desc=desc, map_name=map_name, is_slippery=is_slippery, render_mode="human")
        self.hole_penalty = hole_penalty  # Customize the penalty for falling into a hole

    def step(self, action):
        # Take the standard step in the environment
        state, reward, terminated, truncated, info = super().step(action)

        # Modify the reward if the agent falls into a hole
        if self.desc[state // self.ncol][state % self.ncol] == b'H':
            reward = self.hole_penalty

        return state, reward, terminated, truncated, info


# Register the custom environment
register(
    id="CustomFrozenLake-v0",
    entry_point=__name__ + ":CustomFrozenLakeEnv",  # Reference to the custom class
    kwargs={"map_name": "4x4", "is_slippery": True, "hole_penalty": -1},
)

# Example usage with rendering
if __name__ == "__main__":
    env = gym.make("CustomFrozenLake-v0")  # Create the environment with rendering

    obs, info = env.reset()

    print("Custom FrozenLake Environment with Configurable Hole Behavior")
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()  # Randomly select an action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, State: {obs}, Terminated: {terminated}")

    env.render()
    print("Episode finished!")
    env.close()