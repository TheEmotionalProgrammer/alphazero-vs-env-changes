
from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np
import torch as th




class ObservationEmbedding(ABC):
    observation_space: gym.Space

    def __init__(self, observation_space: gym.Space) -> None:
        self.observation_space = observation_space

    @abstractmethod
    def obs_to_tensor(observation) -> th.Tensor:
        pass

    @abstractmethod
    def obs_dim() -> int:
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


class CoordinateEmbedding(ObservationEmbedding):
    ncols: int
    nrows: int
    observation_space: gym.spaces.Discrete

    def __init__(self, observation_space: gym.spaces.Discrete, *args, ncols=8, **kwargs) -> None:
        super().__init__(observation_space, *args, **kwargs)
        self.ncols = ncols
        self.nrows = observation_space.n // ncols
        print(f"nrows: {self.nrows}, ncols: {self.ncols}")


    def obs_to_tensor(self, observation, *args, **kwargs):
        """
        Returns a tensor of shape (2,) with the coordinates of the observation
        """
        cords = divmod(observation, self.ncols)
        # make cords between -1 and 1
        # cols between 0 and ncols-1, rows between 0 and nrows-1
        cords = (np.array(cords) / np.array([self.nrows-1, self.ncols-1])) * 2 - 1
        return th.tensor(cords, *args, **kwargs)


    def obs_dim(self):
        return 2



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
}
