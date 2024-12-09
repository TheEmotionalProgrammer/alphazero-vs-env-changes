import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from gymnasium.envs.registration import register
from gymnasium.envs.toy_text.utils import categorical_sample


class CustomFrozenLakeEnv(FrozenLakeEnv):
    def __init__(
        self, desc=None, map_name="4x4", is_slippery=False,  hole_reward=-1, terminate_on_hole=False
    ):
        super().__init__(desc=desc, map_name=map_name, hole_reward=hole_reward, is_slippery=is_slippery, render_mode=None)
        self.terminate_on_hole = terminate_on_hole  # Decide if falling into a hole ends the episode

    def step(self, action):
        # Take the standard step in the environment
        transitions = self.P[self.s][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]

        # Check if the new state is a hole
        if self.desc[s // self.ncol][s % self.ncol] == b'H':
            r = self.hole_reward  # Apply the custom hole penalty
            if not self.terminate_on_hole:
                s = self.s  # Stay in the same position
                t = False  # Do not terminate the episode

        self.s = s
        self.lastaction = action

        if self.render_mode == "human":
            self.render()

        return (int(s), r, t, False, {"prob": p})


register(
    id="CustomFrozenLakeNoHoles4x4-v1",
    entry_point=__name__ + ":CustomFrozenLakeEnv",
    kwargs={
        "desc": [
            "SFFF",
            "FFFF",
            "FFFF",
            "FFFG"
            ],
        "map_name": None,
        "is_slippery": False,
        "terminate_on_hole": False,
    },
)

register(
    id="CustomFrozenLakeNoHoles8x8-v1",
    entry_point=__name__ + ":CustomFrozenLakeEnv",
    kwargs={
        "desc": [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFG"
            ],
        "map_name": None,
        "is_slippery": False,
        "terminate_on_hole": False,
    },
)

# Register the custom environment
register(
    id="DefaultFrozenLake4x4-v1",
    entry_point=__name__ + ":CustomFrozenLakeEnv",
    kwargs={
        "map_name": "4x4",
        "is_slippery": False,
        "hole_reward": -1,
        "terminate_on_hole": False,
    },
)

# Register an 8x8 version of the custom environment
register(
    id="DefaultFrozenLake8x8-v1",
    entry_point=__name__ + ":CustomFrozenLakeEnv",
    kwargs={
        "map_name": "8x8",
        "is_slippery": False,
        "hole_reward": -1,
        "terminate_on_hole": False,
    },
)

# Example usage with rendering
if __name__ == "__main__":
    env = gym.make("DefaultFrozenLake8x8-v1", terminate_on_hole=False)  # Set terminate_on_hole=False to test
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