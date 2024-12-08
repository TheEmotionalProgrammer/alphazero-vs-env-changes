from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from sympy import Li, Min
from .utilities.a_star_sp import heuristic, astar_pathfinding, compute_actions_from_path
from .utilities.wrappers import SparseActionsWrapper, gym_wrapper
from minigrid.manual_control import ManualControl
from typing import List, Tuple

from gymnasium import spaces
from gymnasium.core import Wrapper
import numpy as np
    
class ObstaclesGridEnv(MiniGridEnv):

    """
    A simple grid environment with one room and (zero or more) obstacles.
    The obstacles are placed where specified from the user.
    Bumping into an obstacle can result in a negative reward if specified by the user,
    otherwise it just makes the agent stay in the same position with zero reward.
    """

    def __init__(
            self, 
            size=12, 
            agent_start_pos=(1, 6), 
            agent_start_dir=0, 
            max_steps: int | None = None, 
            bump_penalty=0,
            obstacles: List[Tuple[int, int]] | str | None = None, 
        **kwargs):

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

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
        
        self.bump_penalty = bump_penalty
        self.obstacles = obstacles

    @staticmethod   
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square
        self.put_obj(Goal(), width - 2, height - 6)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Place obstacles

        if self.obstacles is not None:

            if isinstance(self.obstacles, list):

                for x, y in self.obstacles:
                    # Check if the position is valid, otherwise raise an error
                    if not x in range(width) or not y in range(height):
                        raise ValueError(f"Obstacle position ({x}, {y}) is out of bounds")
                    
                    self.put_obj(Wall(), x, y)

            elif self.obstacles == "random":
                self.place_rand_obstacles()

        else:
            self.obstacles = []

        self.mission = "grand mission"

    def place_rand_obstacles(self):
        """
        Place obstacles randomly in the grid.
        """
        num_obstacles = np.random.randint(1, 5)
        for _ in range(num_obstacles):
            x, y = self.place_obj(Wall())
            self.obstacles.append((x, y))
              
    def step(self, action):

        """
        Take a step in the environment. Include a negative reward for bumping into walls.
        """
        
        # Determine the intended new position
        fwd_pos = self.front_pos  # Position directly in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)  # Cell in front of the agent

        if action == self.actions.forward:
            # Check if the forward cell is a wall
            if fwd_cell and fwd_cell.type == "wall":
                # Negative reward for bumping into wall
                reward = self.bump_penalty
                self.step_count += 1 # Increment the step count, done in the parent class so we have to do it here if super is not called
                
                return self.gen_obs(), reward, False, False, {}

        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info


def main(policy="random"):
    env =  ImgObsWrapper(FullyObsWrapper(SparseActionsWrapper(ObstaclesGridEnv(render_mode="human"))))

    # Reset the environment
    obs, info = env.reset()
    print("Starting the episode...")

    if policy == "random":

        done = False
        while not done:
            # Randomly select an action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(obs)

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
    
    elif policy == "manual":
        # enable manual control for testing
        manual_control = ManualControl(env, seed=42)
        manual_control.start()



if __name__ == "__main__":
    main("random")