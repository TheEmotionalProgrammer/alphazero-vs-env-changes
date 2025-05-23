from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np
import torch as th
from typing import Dict


class ObservationEmbedding(ABC):
    observation_space: gym.Space

    def __init__(self, observation_space: gym.Space, ncols=None, nrows=None) -> None:
        self.observation_space = observation_space
        self.ncols = ncols
        if nrows is not None:
            self.nrows = nrows
        elif isinstance(observation_space, gym.spaces.Discrete) and ncols is not None:
            self.nrows = observation_space.n // ncols

    @abstractmethod
    def obs_to_tensor(observation) -> th.Tensor:
        pass

    @abstractmethod
    def obs_dim() -> int:
        pass

    @abstractmethod
    def tensor_to_obs(observation) -> th.Tensor:
        pass

class DefaultEmbedding(ObservationEmbedding):

    """
    Embedding that simply turns any observation into a flat tensor.
    """

    def obs_to_tensor(self, observation, *args, **kwargs):
        return th.tensor(
            gym.spaces.flatten(self.observation_space, observation),
            *args,
            **kwargs,
        )

    def obs_dim(self):
        return gym.spaces.flatdim(self.observation_space)

    def tensor_to_obs(self, observation, *args, **kwargs):
        return gym.spaces.unflatten(self.observation_space, observation, *args, **kwargs)

class CoordinateEmbedding(ObservationEmbedding):

    """
    Embedding that turns a discrete observation space into a tensor of shape (2,) with the coordinates of the observation.
    Suitable for grid worlds where the observation is a single integer representing the state.
    """

    ncols: int
    nrows: int
    observation_space: gym.spaces.Discrete
    multiplier: float
    shift: float

    def __init__(self, observation_space: gym.spaces.Discrete, ncols: int, *args, nrows: int | None= None, multiplier = -1.0, shift = 1.0, **kwargs) -> None:
        super().__init__(observation_space, *args, ncols=ncols, nrows=nrows, **kwargs)
        self.multiplier = multiplier
        self.shift = shift


    def obs_to_tensor(self, observation, *args, **kwargs):

        """
        Returns a tensor of shape (2,) with the coordinates of the observation,
        scaled to the range [-1, 1].
        """

        cords = divmod(observation, self.ncols) 
        cords = (np.array(cords) / np.array([self.nrows-1, self.ncols-1])) * self.multiplier + self.shift

        return th.tensor(cords, *args, **kwargs)


    def obs_dim(self):
        return 2

    def tensor_to_obs(self, tensor, *args, **kwargs):

        """
        Returns the discrete observation from a tensor of shape (2,)
        """

        scaled_tensor = (tensor - self.shift) / self.multiplier # First, scale tensor from [-1, 1] to [0, 1]
        scaled_tensor = scaled_tensor * th.tensor([self.nrows-1, self.ncols-1]) # Now scale to [0, nrows-1] for rows and [0, ncols-1] for columns
        indices = scaled_tensor.round().int() # Convert to integer indices
        observation = indices[0] * self.ncols + indices[1] # Convert row and column indices to a single observation index

        return observation
    
class MiniGridEmbedding(ObservationEmbedding):
    width: int
    height: int
    observation_space: gym.spaces.Box

    def __init__(self, observation_space: gym.spaces.Box, width: int, height: int, *args, **kwargs) -> None:
        super().__init__(observation_space, *args, **kwargs)
        self.width = width
        self.height = height

    def obs_to_tensor(self, observation, *args, **kwargs):
        """
        Converts an observation into a tensor.
        
        Returns a tensor [x, y, d] where:
        - x, y: normalized agent coordinates in range [-1, 1].
        - d: normalized agent direction in range [-1, 1].
        """
        # Extract optional arguments to avoid conflict
        dtype = kwargs.pop("dtype", th.float32)

        # Find the agent's position
        agent_pos = np.unravel_index(np.argmax(observation), observation.shape)

        # Find the agent's direction (value at agent's position)
        agent_dir = observation[agent_pos]

        # Normalize position to [-1, 1]
        agent_pos = (np.array(agent_pos) / np.array([self.height - 1, self.width - 1])) * 2 - 1

        # Normalize direction to [-1, 1] (assuming direction is in range [0, 3])
        agent_dir = (agent_dir / 3.0) * 2 - 1

        return th.tensor([*agent_pos, agent_dir], dtype=dtype, *args, **kwargs)

    def obs_dim(self):
        """
        Returns the dimensionality of the tensorized observation.
        """
        return 3

    def tensor_to_obs(self, tensor, *args, **kwargs):
        """
        Converts a tensor [x, y, d] back to the original observation format.
        """
        # Scale tensor from [-1, 1] to [0, 1]
        scaled_tensor = (tensor + 1) / 2.0

        # Scale to [0, height - 1] and [0, width - 1]
        scaled_tensor[:2] *= th.tensor([self.height - 1, self.width - 1])

        # Convert to integer indices for position
        indices = scaled_tensor.round().int()

        # Denormalize direction to range [0, 3]
        agent_dir = (scaled_tensor[2] * 3.0).round().int().item()

        # Construct observation
        observation = np.zeros((self.height, self.width), dtype=np.int32)
        observation[indices[0].item(), indices[1].item()] = agent_dir

        return observation
    
class LunarLanderEmbedding(ObservationEmbedding):
    """
    Embedding for the LunarLander environment.
    Converts the 8-dimensional observation into a tensor.
    """

    observation_space: gym.spaces.Box

    def __init__(self, observation_space: gym.spaces.Box, *args, **kwargs) -> None:
        super().__init__(observation_space, *args, **kwargs)

    def obs_to_tensor(self, observation, *args, **kwargs):
        """
        Converts the LunarLander observation into a tensor.
        The observation is already a flat array, so this simply converts it to a PyTorch tensor.
        """
        # Extract the dtype from kwargs or use a default value
        dtype = kwargs.pop("dtype", th.float32)
        return th.tensor(observation, dtype=dtype, *args, **kwargs)

    def obs_dim(self):
        """
        Returns the dimensionality of the tensorized observation.
        For LunarLander, this is 8.
        """
        return self.observation_space.shape[0]

    def tensor_to_obs(self, tensor, *args, **kwargs):
        """
        Converts a tensor back to the original observation format.
        For LunarLander, this simply converts the tensor back to a NumPy array.
        """
        return tensor.detach().cpu().numpy()
    
class ParkingEmbedding(ObservationEmbedding):
    """
    Embedding for the Parking environment.
    Converts the 6-dimensional observation into a tensor.
    """

    observation_space: Dict

    def __init__(self, observation_space: Dict, *args, **kwargs) -> None:
        super().__init__(observation_space, *args, **kwargs)

    def obs_to_tensor(self, observation, *args, **kwargs):
        """
        Converts the Parking observation into a tensor.
        The observation is already a flat array, so this simply converts it to a PyTorch tensor.
        """

        obs_array = observation["observation"]

        # Extract the dtype from kwargs or use a default value
        dtype = kwargs.pop("dtype", th.float32)
        return th.tensor(obs_array, dtype=dtype, *args, **kwargs)

    def obs_dim(self):
        """
        Returns the dimensionality of the tensorized observation.
        For Parking, this is 6.
        """
        return self.observation_space["observation"].shape[0]

    def tensor_to_obs(self, tensor, *args, **kwargs):
        """
        Converts a tensor back to the original observation format.
        For Parking, this simply converts the tensor back to a NumPy array.
        """
        return tensor.detach().cpu().numpy()
    
class ParkingFullEmbedding(ObservationEmbedding):
    """
    Embedding for the Parking environment.
    Converts the 6-dimensional observation into a tensor.
    """

    observation_space: Dict

    def __init__(self, observation_space: Dict, *args, **kwargs) -> None:
        super().__init__(observation_space, *args, **kwargs)

    def obs_to_tensor(self, observation, *args, **kwargs):
        """
        Converts the Parking observation into a tensor.
        The observation is already a flat array, so this simply converts it to a PyTorch tensor.
        """
        obs, achieved_goal, desired_goal = th.tensor(observation["observation"]), th.tensor(observation["achieved_goal"]), th.tensor(observation["desired_goal"])
        obs_array = th.concat(
            [obs, achieved_goal, desired_goal],
            dim=0,
        )

        # Extract the dtype from kwargs or use a default value
        dtype = kwargs.pop("dtype", th.float32)

        return obs_array.to(dtype)
    
    def obs_dim(self):
        """
        Returns the dimensionality of the tensorized observation.
        """
        return self.observation_space["observation"].shape[0] + self.observation_space["achieved_goal"].shape[0] + self.observation_space["desired_goal"].shape[0]
    
    def tensor_to_obs(self, tensor, *args, **kwargs):

        """
        Converts a tensor back to the original observation format.
        For Parking, this simply converts the tensor back to a NumPy array.
        """
        obs_array = tensor.detach().cpu().numpy()
        return {
            "observation": obs_array[:self.observation_space["observation"].shape[0]],
            "achieved_goal": obs_array[self.observation_space["observation"].shape[0]:self.observation_space["observation"].shape[0] + self.observation_space["achieved_goal"].shape[0]],
            "desired_goal": obs_array[self.observation_space["observation"].shape[0] + self.observation_space["achieved_goal"].shape[0]:],
        }

class ParkingEgoEmbedding(ObservationEmbedding):
    """
    Embedding for the Parking environment that only uses the ego vehicle's observation (first row of the (N, 13) vector).
    """

    def __init__(self, observation_space: gym.Space, *args, **kwargs) -> None:
        super().__init__(observation_space, *args, **kwargs)

    def obs_to_tensor(self, observation, *args, **kwargs):
        """
        Converts only the ego vehicle's observation (obs[0]) into a tensor.
        """
        dtype = kwargs.pop("dtype", th.float32)
        # If observation is a dict, extract the array
        if isinstance(observation, dict):
            obs_array = observation["observation"]
        else:
            obs_array = observation
        ego_obs = np.asarray(obs_array)  # Take only the first row
        # Don't need the last 3 elements 
        return th.tensor(ego_obs, dtype=dtype, *args, **kwargs)

    def obs_dim(self):
        """
        Returns the dimensionality of the ego vehicle's observation.
        """
        # Assumes shape (N, 13)
        # if hasattr(self.observation_space, "shape"):
        #     return self.observation_space.shape[1]
        # Fallback for dict spaces
        return 5

    def tensor_to_obs(self, tensor, *args, **kwargs):
        """
        Converts a tensor back to the ego vehicle's observation (as a numpy array).
        """
        return tensor.detach().cpu().numpy()
        
embedding_dict = {
    "default": DefaultEmbedding,
    "coordinate": CoordinateEmbedding,
    "minigrid": MiniGridEmbedding,
    "lunarlander": LunarLanderEmbedding,
    "parking": ParkingEmbedding,
    "parking_full": ParkingFullEmbedding,
    "parking_ego": ParkingEgoEmbedding,
}
