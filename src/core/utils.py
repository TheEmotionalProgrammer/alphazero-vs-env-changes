import copy
import numpy as np
from environments.lunarlander.lunar_lander import CustomLunarLander

def copy_environment(env):
    """
    Copies the environment. If the environment is CustomLunarLander, uses its proprietary method.
    Otherwise, performs a deep copy.
    """
    if isinstance(env.unwrapped, CustomLunarLander):
        return env.unwrapped.create_copy()
    return copy.deepcopy(env)

def observations_equal(obs1, obs2):

    """
    Compare two observations, handling both scalar and vector cases.
    """
    
    if isinstance(obs1, (int, float)) and isinstance(obs2, (int, float)):
        return obs1 == obs2
    return np.array_equal(obs1, obs2)