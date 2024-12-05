from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from utilities.a_star_sp import heuristic, astar_pathfinding, compute_actions_from_path
from utilities.wrappers import gym_wrapper

from gymnasium import spaces
from gymnasium.core import Wrapper
import numpy as np

class SimpleGridEnv(MiniGridEnv):
    def __init__(
        self,
        size=12,
        agent_start_pos=(1, 6),
        agent_start_dir=0,
        max_steps: int | None = None,
        bump_penalty=0,  # Negative reward for bumping into walls
        terminal_obstacle=False,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.bump_penalty = bump_penalty
        self.terminal_obstacle = terminal_obstacle

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
                
                done = self.terminal_obstacle

                return self.gen_obs(), reward, done, False, {}

        # Default behavior if no wall or lava is bumped into
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info


def main(policy="random"):
    env =  gym_wrapper(SimpleGridEnv(render_mode="human", bump_penalty=-1, terminal_obstacle=True))

    # Reset the environment
    obs, info = env.reset()
    print("Starting the episode...")

    if policy == "random":

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

    elif policy == "a_star":

        path = astar_pathfinding(env)
        actions = compute_actions_from_path(path, env)

        done = False
        for action in actions:
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            print(f"Action: {action}, Reward: {reward}, Done: {done}")

        print("Episode finished!")
        env.close()


if __name__ == "__main__":
    main("random")