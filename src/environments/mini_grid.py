from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper

from gymnasium import spaces
from gymnasium.core import Wrapper
import numpy as np


class UndiscountedRewardWrapper(Wrapper):
    """
        Transform the reward function into a simple:
        - 1 for reaching the goal
        - 0 otherwise

        This is in contrast to the inherent discounting performed by minigrid. 
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if reward > 0:
            reward = 1
        return obs, reward, terminated, truncated, info

class SparseActionsWrapper(Wrapper):
    """
        Reduce the action space to only left, right and forward.
    """
    def __init__(self, env):
        super().__init__(env)
        
        new_action_space = spaces.Discrete(3)
        self.action_space = new_action_space

class SparseFullyObsWrapper(FullyObsWrapper):
    """
        Transform the observation space to have seperate channels for for every dimension of observation.

        The channels correspond to:
        0 - agent location
        1 - agent direction
        2 - wall locations
        3 - goal location

        The observation is also centered around the agent location.
    """
    
    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=1,
            shape=(4, self.env.height, self.env.width),  # number of cells
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        full_grid = self._encode()
        return {**obs, "image": full_grid}

    def _encode(self):
        """
        Produce a sparse numpy encoding of the grid
        """
        env = self.unwrapped 

        # In the original code this is used to handle partial observations
        # here I just set it to all ones
        vis_mask = np.ones((env.grid.width, env.grid.height), dtype=bool)

        observation = np.zeros((self.observation_space['image'].shape[0], env.grid.width, env.grid.height), dtype="uint8")

        # encode the agent location
        # since we will center around the agent location this channel is actually useless
        agent_location = np.array([env.agent_pos[0], env.agent_pos[1]])
        observation[0, agent_location[0], agent_location[1]] = 1

        # agent direction 
        # put a 1 in the location where the agent would be if he moved forward with the current direction
        # (ignoring collisions)
        direction = agent_location + DIR_TO_VEC[env.agent_dir]
        observation[1, direction[0], direction[1]] = 1

        for i in range(env.grid.width):
            for j in range(env.grid.height):
                if vis_mask[i, j]:
                    cell = env.grid.get(i, j)
                    if cell is not None:
                        if cell.type == "wall":
                            observation[2, i, j] = 1
                        if cell.type == "goal":
                            observation[3, i, j] = 1


        # centre everything around the agent location
        x_offset = env.grid.width // 2 - agent_location[0]
        y_offset = env.grid.width // 2 - agent_location[1]
        observation = np.roll(observation, (x_offset, y_offset), axis=(1,2))

        # transpose the x and y coordinates to be better aligned for plotting
        observation = np.transpose(observation[:, :, :], axes=(0,2,1))
        
        return observation


def gym_wrapper(env): 
    return ImgObsWrapper(
                UndiscountedRewardWrapper(
                    SparseActionsWrapper(
                            SparseFullyObsWrapper(env))))

class SimpleGridEnv(MiniGridEnv):
    def __init__(
        self,
        size=12,
        agent_start_pos=(1, 6),
        agent_start_dir=0,
        max_steps: int | None = None,
        bump_penalty=0,  # Negative reward for bumping into walls
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.bump_penalty = bump_penalty

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.grid.set(6, 6, Lava())
        
        # Place a goal square
        self.put_obj(Goal(), width - 2, height - 6)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"

    def step(self, action):
        """
        Take a step in the environment. Include a negative reward for bumping into walls.
        """
        # Determine the intended new position
        fwd_pos = self.front_pos  # Position directly in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)  # Cell in front of the agent

        if action == self.actions.forward:
            # Check if the forward cell is lava
            if fwd_cell and fwd_cell.type == "lava":
                # Negative reward for bumping into lava
                reward = self.bump_penalty
                done = False
                return self.gen_obs(), reward, done, False, {}

        # Default behavior if no wall or lava is bumped into
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info


def main():
    env =  gym_wrapper(SimpleGridEnv(render_mode="human", bump_penalty=-1))

    # Reset the environment
    obs, info = env.reset()
    print("Starting the episode...")

    done = False
    while not done:
        # Randomly select an action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        # Render the environment
        env.render()

        # Print the action and reward for debugging
        print(f"Action: {action}, Reward: {reward}, Done: {done}")

    print("Episode finished!")
    env.close()


if __name__ == "__main__":
    main()