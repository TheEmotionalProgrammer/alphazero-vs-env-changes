import copy
import numpy as np
from environments.lunarlander.lunar_lander import CustomLunarLander, ll_actions_dict
from environments.lunarlander.lunar_lander import VIEWPORT_H, VIEWPORT_W, SCALE, LEG_AWAY, LEG_DOWN
from environments.frozenlake.frozen_lake import CustomFrozenLakeEnv, fz_actions_dict
from environments.highwayenv.parking import ParkingEnv, pk_actions_dict
from environments.highwayenv.parking_new import ParkingSimple 
import gymnasium as gym

def safe_deepcopy_env(obj):
    """
        Perform a deep copy of an environment but without copying its viewer.
    """
    cls = obj.__class__
    result = cls.__new__(cls)
    memo = {id(obj): result}
    for k, v in obj.__dict__.items():
        if k not in ['viewer', '_monitor', 'grid_render', 'video_recorder', '_record_video_wrapper']:
            if isinstance(v, gym.Env):
                setattr(result, k, safe_deepcopy_env(v))
            else:
                setattr(result, k, copy.deepcopy(v, memo=memo))
        else:
            setattr(result, k, None)
    return result

def copy_environment(env):
    """
    Copies the environment. If the environment is CustomLunarLander, uses its proprietary method.
    Otherwise, performs a deep copy.
    """
    if isinstance(env.unwrapped, CustomLunarLander):
        return env.unwrapped.create_copy()
    return safe_deepcopy_env(env.unwrapped)

def observations_equal(obs1, obs2):

    """
    Compare two observations, handling both scalar and vector cases.
    """
    
    if isinstance(obs1, (int, float)) and isinstance(obs2, (int, float)):
        return obs1 == obs2
    return np.array_equal(obs1, obs2)

def actions_dict(env: gym.Env) -> dict:
    """
    Returns the action dictionary for the given environment.
    """
    if isinstance(env.unwrapped, CustomLunarLander):
        return ll_actions_dict
    elif isinstance(env.unwrapped, CustomFrozenLakeEnv):
        return fz_actions_dict
    elif isinstance(env.unwrapped, ParkingEnv):
        return pk_actions_dict
    elif isinstance(env.unwrapped, ParkingSimple):
        return pk_actions_dict

    else:
        raise ValueError(f"Unsupported environment: {env.spec.id}")
    
def print_obs(env: gym.Env, obs):
    """
    Prints the observation in a human-readable format.
    """

    if obs is None:
        return "None"

    if isinstance(env.unwrapped, CustomLunarLander):
        posx, posy = float(obs[0]), float(obs[1])

        # Truncate to 5 decimal places
        # posx = float(str(round(posx, 3)))
        # posy = float(str(round(posy, 3)))

        return obs
        
    elif isinstance(env.unwrapped, CustomFrozenLakeEnv):
        ncols = env.unwrapped.ncol
        return obs // ncols, obs % ncols
    elif isinstance(env.unwrapped, ParkingEnv):
        obs = obs["observation"]
        return obs
    elif isinstance(env.unwrapped, ParkingSimple):
        return "A"
    else:
        print("Unknown Observation Type")