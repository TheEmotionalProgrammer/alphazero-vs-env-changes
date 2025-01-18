import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from gymnasium.envs.registration import register
from gymnasium.envs.toy_text.utils import categorical_sample
from matplotlib import pyplot as plt
import sys

sys.path.append("src/")
from log_code.gen_renderings import save_gif_imageio

actions_dict = {
    0: "Left",
    1: "Down",
    2: "Right",
    3: "Up",
}


class CustomFrozenLakeEnv(FrozenLakeEnv):
    def __init__(
        self, desc=None, map_name="4x4", is_slippery=False,  hole_reward=0, terminate_on_hole=False, render_mode=None, deviation_type = "bump"
    ):
        super().__init__(desc=desc, map_name=map_name, hole_reward=hole_reward, is_slippery=is_slippery, render_mode=render_mode)
        self.terminate_on_hole = terminate_on_hole  # Decide if falling into a hole ends the episode
        self.deviation_type = deviation_type

    def step(self, action):
        # Take the standard step in the environment
        transitions = self.P[self.s][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]

        if self.desc[s // self.ncol][s % self.ncol] == b'H':
            r = self.hole_reward  # Apply the custom hole penalty
            if not self.terminate_on_hole:
                if self.deviation_type == "bump":
                    s = self.s  # Stay in the same position
                elif self.deviation_type in ["clockwise", "counter_clockwise"]:
                    # Define correct mappings for clockwise and counterclockwise deviations
                    if self.deviation_type == "clockwise":
                        action_map = {0: 3, 3: 2, 2: 1, 1: 0}  # Correct clockwise mapping
                    elif self.deviation_type == "counter_clockwise":
                        action_map = {0: 1, 1: 2, 2: 3, 3: 0}  # Correct counterclockwise mapping
                    
                    # Update the action based on the deviation
                    new_action = action_map[action]

                    # Try stepping in the new direction
                    transitions = self.P[self.s][new_action]
                    i = categorical_sample([t[0] for t in transitions], self.np_random)
                    p, s, r, t = transitions[i]  # Update state, reward, and terminal status

                    if self.desc[s // self.ncol][s % self.ncol] == b'H': # If would you encounter an obstacle again, just stay in the same position
                        r = self.hole_reward
                        s = self.s  # Stay in the same position
                else:
                    raise ValueError(f"Invalid deviation type: {self.deviation_type}")

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
        "map_name": None,
        "is_slippery": False,
        "hole_reward": -1,
        "terminate_on_hole": False,
    },
)

# Example usage with rendering
if __name__ == "__main__":
    frames = []
    env = gym.make("DefaultFrozenLake8x8-v1", terminate_on_hole=False, render_mode = "rgb_array")  # Set terminate_on_hole=False to test

    obs, info = env.reset()

    if env.unwrapped.render_mode == "rgb_array":
        frames.append(env.render())

    print("Custom FrozenLake Environment with Configurable Hole Behavior")
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if env.unwrapped.render_mode == "rgb_array":
            frames.append(env.render())
        print(f"Action: {action}, Reward: {reward}, State: {obs}, Terminated: {terminated}")

    if env.unwrapped.render_mode == "rgb_array":
        save_gif_imageio(frames, output_path="output.gif", fps=5)

    print("Episode finished!")
    env.close()