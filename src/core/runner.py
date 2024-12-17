import copy
import multiprocessing

from tensordict import TensorDict
import torch as th
import gymnasium as gym
import numpy as np
from core.mcts import MCTS
from environments.observation_embeddings import ObservationEmbedding
from policies.policies import PolicyDistribution, custom_softmax
from environments.frozenlake.frozen_lake import actions_dict
from core.node import Node
def run_episode_process(args):
    """Wrapper function for multiprocessing that unpacks arguments and runs a single episode."""
    return run_episode(*args)

def collect_trajectories(tasks, workers=1):
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            # Run the tasks using map
            results = pool.map(run_episode_process, tasks)
    else:
        results = [run_episode_process(task) for task in tasks]
    res_tensor =  th.stack(results)
    return res_tensor

@th.no_grad()
def run_episode(
    solver: MCTS,
    env: gym.Env,
    tree_evaluation_policy: PolicyDistribution,
    observation_embedding: ObservationEmbedding,
    planning_budget=1000, # number of simulations to run in the planning tree (at each step)
    max_steps=1000, # maximum number of steps to take in the (real) environment
    seed=None,
    temperature=None,
    azdetection=False,
    original_env: gym.Env | None = None,
    unroll_steps=5,
    return_trees=False,
    ):

    """
    Runs an episode using the given solver and environment.
    We step into the (real) enviroment for a maximum of max_steps steps, or until the environment terminates.
    For each timestep, the trajectory contains the observation, the policy distribution, the action taken and the reward received.
    Outputs the trajectory and optionally the trees that were generated during the episode.
    """

    assert isinstance(env.action_space, gym.spaces.Discrete)
    n = int(env.action_space.n)

    if seed is not None:
        th.manual_seed(seed)
        np.random.seed(seed)

    observation, info = env.reset(seed=seed)

    old_obs = observation

    if original_env is not None:
        _, _ = original_env.reset(seed=seed) 
        original_env.unwrapped.s = env.unwrapped.s

    observation_tensor: th.Tensor = observation_embedding.obs_to_tensor(observation, dtype=th.float32)
    trajectory = TensorDict(
        source={
            "observations": th.zeros(
                max_steps,
                observation_embedding.obs_dim(),
                dtype=observation_tensor.dtype,
            ),
            "rewards": th.zeros(max_steps, dtype=th.float32),
            "policy_distributions": th.zeros(max_steps, n, dtype=th.float32),
            "actions": th.zeros(max_steps, dtype=th.int64),
            "mask": th.zeros(max_steps, dtype=th.bool),
            "terminals": th.zeros(max_steps, dtype=th.bool),
            "root_values": th.zeros(max_steps, dtype=th.float32),
        }, 
        batch_size=[max_steps],
    )
    if return_trees:
        trees = []

    if azdetection:

        tree = solver.search(env,planning_budget, observation, 0.0, original_env = original_env, n=unroll_steps)
        
    else:

        tree = solver.search(env,planning_budget, observation, 0.0)

    step = 0

    while step < max_steps:
        
        if azdetection and solver.planning_style == "value_search":
            # Check if tree is a list
            if isinstance(tree, list):
                # If it is, then it is a list of actions that the agent have to take
                for action in tree:
                    
                    new_obs, reward, terminated, truncated, _ = env.step(action)
                    new_pos_row = new_obs // 8
                    new_pos_col = new_obs % 8
                    print(f"obs = ({new_pos_row}, {new_pos_col}), reward = {reward}, terminated = {terminated}, truncated = {truncated}")
                    if original_env is not None:
                        if new_obs != old_obs:
                            _, _, _, _, _ = original_env.step(action)
                    old_obs = new_obs
                    assert not truncated
                    next_terminal = terminated

                    trajectory["observations"][step] = observation_tensor
                    trajectory["rewards"][step] = reward
                    #trajectory["policy_distributions"][step] = policy_dist.probs
                    trajectory["actions"][step] = action
                    trajectory["mask"][step] = True
                    trajectory["terminals"][step] = next_terminal
                    #trajectory["root_values"][step] = th.tensor(root_value, dtype=th.float32)
                    if next_terminal or truncated:
                        break
                    new_observation_tensor = observation_embedding.obs_to_tensor(new_obs, dtype=th.float32)
                    observation_tensor = new_observation_tensor
                    step += 1
                
                print("Here")
                tree = Node (
                    observation = new_obs,
                    parent = None,
                    env = env,
                    terminal = terminated,
                    reward = reward,
                    action_space=env.action_space,
                )

                tree.value_evaluation = solver.value_function(tree)
                solver.backup(tree, tree.value_evaluation)
                       

        root_value = tree.value_evaluation # Contains the value estimate of the root node computed by the planning step

        tree.reset_var_val()

        policy_dist = tree_evaluation_policy.softmaxed_distribution(tree) # Evaluates the tree using the given evaluation policy (e.g., visitation counts)

        if return_trees:
            trees.append(tree)

        if not azdetection:
            # apply extra softmax
            action = th.distributions.Categorical(probs=custom_softmax(policy_dist.probs, temperature, None)).sample().item()
            # res will now contain the observation, policy distribution, action, as well as the reward and terminal we got from executing the action

        else:

            # Check if the unrolling has detected a problem

            if solver.problem_idx is None:
                
                print("No problem detected, acting normally.")

                action = th.argmax(solver.model.single_observation_forward(tree.observation)[1]).item()
            
            else:

                #print(f"Problem detected at step {solver.problem_idx}, following the planning tree")
                # apply extra softmax

                action = th.distributions.Categorical(probs=custom_softmax(policy_dist.probs, temperature, None)).sample().item()
                # res will now contain the observation, policy distribution, action, as well as the reward and terminal we got from executing the action

            print(f"action = {actions_dict[action]}")

        new_obs, reward, terminated, truncated, _ = env.step(action)

        new_pos_row = new_obs // 8 # Convert the observation to a 2D position, hardcoded for now
        new_pos_col = new_obs % 8

        print(f"obs = ({new_pos_row}, {new_pos_col}), reward = {reward}, terminated = {terminated}, truncated = {truncated}")

        if original_env is not None:

            if new_obs != old_obs:
            
                _, _, _, _, _ = original_env.step(action)

        old_obs = new_obs
        
        assert not truncated

        next_terminal = terminated
        trajectory["observations"][step] = observation_tensor
        trajectory["rewards"][step] = reward
        trajectory["policy_distributions"][step] = policy_dist.probs
        trajectory["actions"][step] = action
        trajectory["mask"][step] = True
        trajectory["terminals"][step] = next_terminal
        trajectory["root_values"][step] = th.tensor(root_value, dtype=th.float32)
        if next_terminal or truncated:
            break
        
        if original_env is not None:
            tree = solver.search(env, planning_budget, new_obs, reward, original_env=original_env, n=unroll_steps) # Computes a planning tree using the given solver and available budget. Returns the root node of the tree.

        else:
            tree = solver.search(env, planning_budget, new_obs, reward)

        new_observation_tensor = observation_embedding.obs_to_tensor(new_obs, dtype=th.float32)
        observation_tensor = new_observation_tensor

        step += 1

    # if we terminated early, we need to add the final observation to the trajectory as well for value estimation
    # trajectory.append((observation, None, None, None, None))
    # observations.append(observation)
    # convert render to tensor

    if return_trees:
        return trajectory, trees

    return trajectory
