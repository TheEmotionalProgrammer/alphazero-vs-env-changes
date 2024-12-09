
from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np
import torch as th


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
        Returns a tensor of shape (2,) with the coordinates of the observation
        """
        cords = divmod(observation, self.ncols)
        # make cords between -1 and 1
        # cols between 0 and ncols-1, rows between 0 and nrows-1
        cords = (np.array(cords) / np.array([self.nrows-1, self.ncols-1])) * self.multiplier + self.shift
        return th.tensor(cords, *args, **kwargs)


    def obs_dim(self):
        return 2

    def tensor_to_obs(self, tensor, *args, **kwargs):
        """
        Returns the observation from a tensor of shape (2,)
        """
        # First, scale tensor from [-1, 1] to [0, 1]
        scaled_tensor = (tensor - self.shift) / self.multiplier
        # Now scale to [0, nrows-1] for rows and [0, ncols-1] for columns
        scaled_tensor = scaled_tensor * th.tensor([self.nrows-1, self.ncols-1])
        # Convert to integer indices
        indices = scaled_tensor.round().int()
        # Convert row and column indices to a single observation index
        observation = indices[0] * self.ncols + indices[1]
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



        





class TaxiEmbedding(ObservationEmbedding):
    passenger_locations = 5
    destinations = 4
    ncols = 5

    def obs_to_tensor(self, observation, *args, **kwargs):
        """
        Convert the observation to a tensor
        - One hot encode the passenger location and destination
        - Coordinate encode the taxi location
        - An observation is returned as an int() that encodes the corresponding state, calculated by
          ((taxi_row * ncols + taxi_col) * passenger_locations + passenger_location) * destinations + destination
        """
        # Decode the observation into its components
        destination = observation % self.destinations
        intermediate = observation // self.destinations
        passenger_location = intermediate % self.passenger_locations
        intermediate = intermediate // self.passenger_locations
        taxi_col = intermediate % self.ncols
        taxi_row = intermediate // self.ncols

        # One-hot encode the passenger location and destination
        passenger_location_tensor = th.zeros(self.passenger_locations, *args, **kwargs)
        passenger_location_tensor[passenger_location] = 1
        destination_tensor = th.zeros(self.destinations, *args, **kwargs)
        destination_tensor[destination] = 1

        # Coordinate encode the taxi location (simply use the numerical values here)
        taxi_location_tensor = th.tensor([taxi_row, taxi_col], *args, **kwargs)

        # Combine all tensors into a single tensor
        # Note: This step might vary based on your specific needs for input shape
        combined_tensor = th.cat([taxi_location_tensor, passenger_location_tensor, destination_tensor])

        return combined_tensor


    def obs_dim(self):
        return 2 + self.passenger_locations + self.destinations


embedding_dict = {
    "default": DefaultEmbedding,
    "coordinate": CoordinateEmbedding,
    "taxi": TaxiEmbedding,
    "minigrid": MiniGridEmbedding,
}
