from gymnasium import spaces
import gymnasium as gym
from gymnasium.core import Wrapper
import numpy as np
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, ObservationWrapper
from minigrid.core.constants import DIR_TO_VEC

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

class CoordinateFullyObsWrapper(ObservationWrapper):

    """
    Only return the agent's position and direction in the observation.
    """

    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=4,  # 4 directions
            shape=(self.env.unwrapped.width, self.env.unwrapped.height, 1),  # number of cells
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):

        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir

        # Initialize the observation image
        image = np.zeros((self.env.unwrapped.height, self.env.unwrapped.width), dtype="uint8")
        
        # Mark the agent's position
        image[agent_pos[1], agent_pos[0]] = agent_dir + 1  # Adding 1 to differentiate from zeros

        return {**obs, "image": image}
    
def gym_wrapper(env: gym.Env) -> gym.Env: 
    return ImgObsWrapper(
                UndiscountedRewardWrapper(
                    SparseActionsWrapper(
                            CoordinateFullyObsWrapper(env))))
