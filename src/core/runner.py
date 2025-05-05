import multiprocessing
import os
import subprocess

from tensordict import TensorDict
import torch as th
import gymnasium as gym
import numpy as np
from core.mcts import MCTS, RandomRolloutMCTS, NoLoopsMCTS
from az.azmcts import AlphaZeroMCTS
from azdetection.minitrees import MiniTrees
from azdetection.megatree import MegaTree
from azdetection.pddp import PDDP
from environments.observation_embeddings import ObservationEmbedding
from environments.lunarlander.lunar_lander import CustomLunarLander
from environments.frozenlake.frozen_lake import CustomFrozenLakeEnv
from policies.policies import PolicyDistribution, custom_softmax
from policies.tree_policies import MinimalVarianceConstraintPolicyPrior
from policies.utility_functions import policy_value, policy_value_variance
from core.node import Node
from core.utils import copy_environment, observations_equal, actions_dict, print_obs
import matplotlib.pyplot as plt


import copy

from log_code.gen_renderings import save_gif_imageio

from policies.utility_functions import get_children_visits


def collect_trajectories(tasks, workers=1):
    subprocess.run(["pwd"])
    if workers > 1:
        print(os.getcwd())
        with multiprocessing.Pool(workers) as pool:
            # Run the tasks using map
            results = pool.map(run_episode_process, tasks)
    else:
        results = [run_episode_process(task) for task in tasks] 

    # check if the results are tuples, if so, unpack them
    if all(isinstance(result, tuple) for result in results):
        trajectories, trees = zip(*results)
        res_tensor = th.stack(trajectories)
        return res_tensor, trees
    else:
        res_tensor =  th.stack(results)
        return res_tensor

def run_episode_process(args):

    """Wrapper function for multiprocessing that unpacks arguments and runs a single episode with the specified algorithm."""

    agent = args[0]

    if isinstance(agent, MegaTree):
        return run_episode_megatree(*args)
    
    elif isinstance(agent, MiniTrees):
        return run_episode_minitrees(*args)
    
    elif isinstance(agent, PDDP):
        return run_episode_pddp(*args)
    
    elif isinstance(agent, NoLoopsMCTS):
        return run_episode_no_loop(*args)
    
    elif isinstance(agent, AlphaZeroMCTS) or isinstance(agent, RandomRolloutMCTS):
        return run_episode_azmcts(*args)
    
    
@th.no_grad()
def run_episode_azmcts(
    solver: MCTS,
    env: gym.Env,
    tree_evaluation_policy: PolicyDistribution,
    observation_embedding: ObservationEmbedding,
    planning_budget=1000,
    max_steps=1000,
    seed=None,
    temperature=None,
    original_env: gym.Env | None = None,
    unroll_steps=5,
    render=False,
    return_trees=False,
):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    n = int(env.action_space.n)

    if seed is not None:
        th.manual_seed(seed)
        np.random.seed(seed)

    observation, _ = env.reset(seed=seed)

    print(f"Env: obs = {print_obs(env, observation)}")

    if render:
        vis_env = copy_environment(env)  # Use the utility function
        vis_env.unwrapped.render_mode = "rgb_array"
        frames = [vis_env.render()]

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

    tree = solver.search(env, planning_budget, observation, 0.0)

    step = 0

    while step < max_steps:
        root_value = tree.value_evaluation
        #tree.reset_var_val()
        policy_dist = tree_evaluation_policy.softmaxed_distribution(tree)

        #print(f"Step {step}: {policy_dist.probs}")

        if return_trees:
            tree_copy = copy.deepcopy(tree)
            trees.append(tree_copy)

        distribution = th.distributions.Categorical(probs=custom_softmax(policy_dist.probs, temperature, None))

        action = distribution.sample().item()

        print(f"Env: action = {actions_dict(env)[action]}")

        new_obs, reward, terminated, truncated, _ = env.step(action)

        print(f"Env: step = {step}, obs = {print_obs(env, new_obs)}, reward = {reward}, terminated = {terminated}, truncated = {truncated}")

        if render:
            vis_env.step(action)
            frames.append(vis_env.render())

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

        tree = solver.search(env, planning_budget, new_obs, reward)

        new_observation_tensor = observation_embedding.obs_to_tensor(new_obs, dtype=th.float32)
        observation_tensor = new_observation_tensor

        step += 1

    if render:
        fps = 5
        if isinstance(env.unwrapped, CustomLunarLander):
            fps = 30
        save_gif_imageio(frames, output_path=f"gifs/output.gif", fps=fps)

    if return_trees:
        return trajectory, trees

    return trajectory

@th.no_grad()
def run_episode_no_loop(
    solver: MCTS,
    env: gym.Env,
    tree_evaluation_policy: PolicyDistribution,
    observation_embedding: ObservationEmbedding,
    planning_budget=1000,
    max_steps=1000,
    seed=None,
    temperature=None,
    original_env: gym.Env | None = None,
    unroll_steps=5,
    render=False,
    return_trees=False,
):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    n = int(env.action_space.n)

    if seed is not None:
        th.manual_seed(seed)
        np.random.seed(seed)

    observation, _ = env.reset(seed=seed)

    print(f"Env: obs = {print_obs(env, observation)}")

    if render:
        vis_env = copy_environment(env)  # Use the utility function
        vis_env.unwrapped.render_mode = "rgb_array"
        frames = [vis_env.render()]

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

    tree = solver.search(env, planning_budget, observation, 0.0)

    step = 0

    while step < max_steps:
        root_value = tree.value_evaluation
        #tree.reset_var_val()
        policy_dist = tree_evaluation_policy.softmaxed_distribution(tree, action_mask=tree.mask)

        #print(f"Step {step}: {policy_dist.probs}")

        if return_trees:
            tree_copy = copy.deepcopy(tree)
            trees.append(tree_copy)

        distribution = th.distributions.Categorical(probs=custom_softmax(policy_dist.probs, temperature, None))

        action = distribution.sample().item()

        #print(f"Env: action = {actions_dict(env)[action]}")

        new_obs, reward, terminated, truncated, _ = env.step(action)

        print(f"Env: step = {step}, obs = {print_obs(env, new_obs)}, reward = {reward}, terminated = {terminated}, truncated = {truncated}")

        if render:
            vis_env.step(action)
            frames.append(vis_env.render())

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

        tree = solver.search(env, planning_budget, new_obs, reward)

        new_observation_tensor = observation_embedding.obs_to_tensor(new_obs, dtype=th.float32)
        observation_tensor = new_observation_tensor

        step += 1

    if render:
        fps = 5
        if isinstance(env.unwrapped, CustomLunarLander):
            fps = 30
        save_gif_imageio(frames, output_path=f"gifs/output.gif", fps=fps)

    if return_trees:
        return trajectory, trees

    return trajectory


@th.no_grad()
def run_episode_pddp(
    solver: MCTS,
    env: gym.Env,
    tree_evaluation_policy: PolicyDistribution,
    observation_embedding: ObservationEmbedding,
    planning_budget=1000, # number of simulations to run in the planning tree (at each step)
    max_steps=1000, # maximum number of steps to take in the (real) environment
    seed=None,
    temperature=None,
    original_env: gym.Env | None = None, # Used to keep track of the original environment state for detection purposes in an ideal context
    unroll_steps=5,
    render=False,
    return_trees=False,
):
    
    """
    Runs an episode using the given solver and environment.
    We step into the (real) enviroment for a maximum of max_steps steps, or until the environment terminates.
    For each timestep, the trajectory contains the observation, the policy distribution, the action taken and the reward received.
    Outputs the trajectory and optionally the trees that were generated during the episode.
    """

    assert isinstance(env.action_space, gym.spaces.Discrete) # For now, only supports discrete action spaces
    n = int(env.action_space.n)

    if seed is not None:
        th.manual_seed(seed)
        np.random.seed(seed)

    observation, info = env.reset(seed=seed)

    print(f"Env: obs = {print_obs(env, observation)}")

    if render:
        if isinstance(env.unwrapped, CustomLunarLander):
            vis_env = env.unwrapped.create_copy()
        else:
            vis_env = copy.deepcopy(env) # Used to visualize the environment in the case of the frozenlake

        vis_env.unwrapped.render_mode = "rgb_array"
        frames = [vis_env.render()]

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
    
    tree = solver.search(env,planning_budget, observation, 0.0)

    step = 0
    
    while step < max_steps:
        
        root_value = tree.value_evaluation # Contains the value estimate of the root node computed by the planning step

        #tree.reset_var_val() # The value and variance (mvc) estimates of the whole subtree are reset.

        #print(tree.prior_policy)

        policy_dist = tree_evaluation_policy.softmaxed_distribution(tree) # Evaluates the tree using the given evaluation policy (e.g., visitation counts)

        if return_trees:
            tree.policy_value = policy_value(tree, tree_evaluation_policy, solver.discount_factor)
            tree.variance = policy_value_variance(tree, tree_evaluation_policy, solver.discount_factor)
            tree_copy = copy.deepcopy(tree) 
            trees.append(tree_copy)
        
        distribution = th.distributions.Categorical(probs=custom_softmax(policy_dist.probs, temperature, None)) # apply extra softmax

        action = distribution.sample().item() # Note that if the temperature of the softmax was zero, this becomes an argmax

        new_obs, reward, terminated, truncated, _ = env.step(action)

        if render:
            vis_env.step(action)
            frames.append(vis_env.render())

        print(f"Env: step = {step}, obs = {print_obs(env, new_obs)}, reward = {reward}, terminated = {terminated}, truncated = {truncated}")
        
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

        tree = solver.search(env, planning_budget, new_obs, reward, lastaction=action)

        new_observation_tensor = observation_embedding.obs_to_tensor(new_obs, dtype=th.float32)
        observation_tensor = new_observation_tensor

        step += 1
    
    if render:
        fps = 5
        if isinstance(env.unwrapped, CustomLunarLander):
            fps = 30
        save_gif_imageio(frames, output_path=f"gifs/output.gif", fps=fps)

    if return_trees:
        return trajectory, trees

    return trajectory
    
def run_episode_minitrees(
    solver: MCTS,
    env: gym.Env,
    tree_evaluation_policy: PolicyDistribution,
    observation_embedding: ObservationEmbedding,
    planning_budget=1000, # number of simulations to run in the planning tree (at each step)
    max_steps=1000, # maximum number of steps to take in the (real) environment
    seed=None,
    temperature=None,
    original_env: gym.Env | None = None, # Used to keep track of the original environment state for detection purposes in an ideal context 
    unroll_steps=5,
    render=False,
    return_trees=False,
):
    """
    Runs an episode using the given solver and environment.
    We step into the (real) enviroment for a maximum of max_steps steps, or until the environment terminates.
    For each timestep, the trajectory contains the observation, the policy distribution, the action taken and the reward received.
    Outputs the trajectory and optionally the trees that were generated during the episode.
    """

    total_planning = 0

    assert isinstance(env.action_space, gym.spaces.Discrete) # For now, only supports discrete action spaces
    n = int(env.action_space.n)

    if seed is not None:
        th.manual_seed(seed)
        np.random.seed(seed)

    observation, info = env.reset(seed=seed)

    if render:
        vis_env = copy_environment(env)  # Use the utility function
        vis_env.unwrapped.render_mode = "rgb_array"
        frames = [vis_env.render()]

    old_obs = observation

    if original_env is not None:
        _, _ = original_env.reset(seed=seed) 
        original_env.unwrapped.s = env.unwrapped.s # Set the initial state of the original environment to match the current environment

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

    tree, net_planning = solver.search(env,planning_budget, observation, 0.0, original_env = original_env, n=unroll_steps, last_action=None)

    #print(f"Net planning step 0: {net_planning}")

    total_planning += net_planning

    step = 0

    while step < max_steps:

        if solver.value_search and isinstance(tree, list):
            """
            If the agent is using the value search planning style and the search returned a list of actions,
            then we want to just follow those actions.
            """
            close = False 
            for action in tree:

                new_obs, reward, terminated, truncated, _ = env.step(action)

                if render:
                    vis_env.step(action)
                    frames.append(vis_env.render())

                print(f"obs = {print_obs(env, new_obs)}")

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
                    close = True
                    break
                new_observation_tensor = observation_embedding.obs_to_tensor(new_obs, dtype=th.float32)
                observation_tensor = new_observation_tensor

                step += 1 # Still need to increment the step counter 
                if step >= max_steps:
                    close=True
                    break

            if close:
                break

            tree = Node ( # Create a new node to represent the current state after following the actions
                observation = new_obs,
                parent = None,
                env = env,
                terminal = terminated,
                reward = reward,
                action_space=env.action_space,
            )

            # Note that despite in principle we now act without planning for the next step, 
            # in practice if the detection said now the path is clear we'll just follow the prior,
            # so it wouldn't make sense to plan calling solver.search() here.

            # We still compute the standard value eval to avoid breaking code but this shouldn't be needed
            tree.value_evaluation = solver.value_function(tree)

            solver.backup(tree, tree.value_evaluation)
                       
        root_value = tree.value_evaluation # Contains the value estimate of the root node computed by the planning step

        #tree.reset_var_val() # The value and variance (mvc) estimates of the whole subtree are reset.

        policy_dist = tree_evaluation_policy.softmaxed_distribution(tree) # Evaluates the tree using the given evaluation policy (e.g., visitation counts)

        if return_trees:
            tree_copy = copy.deepcopy(tree) 
            trees.append(tree_copy)

        # if solver.problem_idx is None: # If no problem was detected, we act following the prior (quick)
        #     #print("No problem detected, acting normally.")
        action = th.argmax(tree.prior_policy).item()

        # else: # If a problem was detected, we act following the policy distribution

        #     #print("Problem detected, acting according to the policy distribution.")

        #     distribution = th.distributions.Categorical(probs=custom_softmax(policy_dist.probs, temperature, None)) # apply extra softmax

        #     # Check if any children has zero visits, or if any logit is equal to another logit (tie)
        #     if th.any(get_children_visits(tree) == 0) or th.equal(th.max(policy_dist.logits), th.min(policy_dist.logits)):
        #         #print("Not enough visits or tie, following the prior")
        #         action = th.argmax(tree.prior_policy).item()

        #     # else:
        #     action = distribution.sample().item() # Note that if the temperature of the softmax was zero, this becomes an argmax

        #     print(f"Env: action = {actions_dict(env)[action]}")

        new_obs, reward, terminated, truncated, _ = env.step(action)

        if render:
            vis_env.step(action)
            frames.append(vis_env.render())

        # Convert the observation to a 2D position, hardcoded size of the grid for now
        # new_pos_row = new_obs // observation_embedding.ncols
        # new_pos_col = new_obs % observation_embedding.ncols

        #print(f"Env: obs = ({new_pos_row}, {new_pos_col}), reward = {reward}, terminated = {terminated}, truncated = {truncated}")

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


        tree, net_planning = solver.search(env, planning_budget, new_obs, reward, original_env=original_env, n=unroll_steps, last_action=action) 

        #print(f"Planning step {step+1}: {net_planning}")
        total_planning += net_planning

        new_observation_tensor = observation_embedding.obs_to_tensor(new_obs, dtype=th.float32)
        observation_tensor = new_observation_tensor

        step += 1

    if render:
        fps = 5
        if isinstance(env.unwrapped, CustomLunarLander):
            fps = 30
        save_gif_imageio(frames, output_path=f"gifs/output.gif", fps=fps)

    if return_trees:
        return trajectory, trees

    print(f"Total planning steps: {total_planning}")
    print("Average planning steps per action: ", total_planning/step)

    return trajectory

@th.no_grad()
def run_episode_megatree(
    solver: MCTS,
    env: gym.Env,
    tree_evaluation_policy: PolicyDistribution,
    observation_embedding: ObservationEmbedding,
    planning_budget=1000, # number of simulations to run in the planning tree (at each step)
    max_steps=1000, # maximum number of steps to take in the (real) environment
    seed=None,
    temperature=None,
    original_env: gym.Env | None = None, # Used to keep track of the original environment state for detection purposes in an ideal context 
    unroll_steps=5,
    render=False,
    return_trees=False,
    ):

    """
    Runs an episode using the given solver and environment.
    We step into the (real) enviroment for a maximum of max_steps steps, or until the environment terminates.
    For each timestep, the trajectory contains the observation, the policy distribution, the action taken and the reward received.
    Outputs the trajectory and optionally the trees that were generated during the episode.
    """

    assert isinstance(env.action_space, gym.spaces.Discrete) # For now, only supports discrete action spaces
    n = int(env.action_space.n)

    if seed is not None:
        th.manual_seed(seed)
        np.random.seed(seed)

    observation, info = env.reset(seed=seed)

    if render:
        vis_env = copy.deepcopy(env) # Used to visualize the environment in the case of the frozenlake
        vis_env.unwrapped.render_mode = "rgb_array"
        frames = [vis_env.render()]

    old_obs = observation

    if original_env is not None:
        _, _ = original_env.reset(seed=seed) 
        original_env.unwrapped.s = env.unwrapped.s # Set the initial state of the original environment to match the current environment

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
    
    total_planning = 0

    tree, net_planning = solver.search(env,planning_budget, observation, 0.0, original_env = original_env, n=unroll_steps, env_action=None)

    print(f"Net planning step 0: {net_planning}")
    
    total_planning += net_planning

    step = 0
    
    while step < max_steps:
        
        if solver.value_search and isinstance(tree, list):
            """
            If the agent is using the value search planning style and the search returned a list of actions,
            then we want to just follow those actions.
            """
            close = False 
            for action in tree:
                
                new_obs, reward, terminated, truncated, _ = env.step(action)

                if render:
                    vis_env.step(action)
                    frames.append(vis_env.render())

                new_pos_row = new_obs // observation_embedding.ncols
                new_pos_col = new_obs % observation_embedding.ncols
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
                    close = True
                    break
                new_observation_tensor = observation_embedding.obs_to_tensor(new_obs, dtype=th.float32)
                observation_tensor = new_observation_tensor

                step += 1 # Still need to increment the step counter 
                if step >= max_steps:
                    close=True
                    break
            
            if close:
                break

            tree = Node ( # Create a new node to represent the current state after following the actions
                observation = new_obs,
                parent = None,
                env = env,
                terminal = terminated,
                reward = reward,
                action_space=env.action_space,
            )

            # Note that despite in principle we now act without planning for the next step, 
            # in practice if the detection said now the path is clear we'll just follow the prior,
            # so it wouldn't make sense to plan calling solver.search() here.

            # We still compute the standard value eval to avoid breaking code but this shouldn't be needed
            tree.value_evaluation = solver.value_function(tree)

            solver.backup(tree, tree.value_evaluation)

            solver.problem_idx = None # Reset the problem index
            solver.stop_unrolling = False # Reset the stop unrolling flag
            solver.trajectory = [] # Reset the trajectory 

        root_value = tree.value_evaluation # Contains the value estimate of the root node computed by the planning step

        #tree.reset_var_val() # The value and variance (mvc) estimates of the whole subtree are reset.

        policy_dist = tree_evaluation_policy.softmaxed_distribution(tree) # Evaluates the tree using the given evaluation policy (e.g., visitation counts)

        if return_trees:
            tree_copy = copy.deepcopy(tree) 
            trees.append(tree_copy)

        action = th.argmax(tree.prior_policy).item()

        new_obs, reward, terminated, truncated, _ = env.step(action)

        if render:
            vis_env.step(action)
            frames.append(vis_env.render())

        # Convert the observation to a 2D position, hardcoded size of the grid for now
        new_pos_row = new_obs // observation_embedding.ncols
        new_pos_col = new_obs % observation_embedding.ncols

        print(f"Env: obs = ({new_pos_row}, {new_pos_col}), reward = {reward}, terminated = {terminated}, truncated = {truncated}")

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

        tree, net_planning = solver.search(env, planning_budget, new_obs, reward, original_env=original_env, n=unroll_steps, env_action=action) 
        print(f"Planning step {step+1}: {net_planning}")
        total_planning += net_planning
        new_observation_tensor = observation_embedding.obs_to_tensor(new_obs, dtype=th.float32)
        observation_tensor = new_observation_tensor

        step += 1
    
    print(f"Total planning steps: {total_planning}")
    print("Average planning steps per action: ", total_planning/step)
    
    if render:
        fps = 5
        if isinstance(env.unwrapped, CustomLunarLander):
            fps = 30
        save_gif_imageio(frames, output_path=f"gifs/output.gif", fps=fps)

    if return_trees:
        return trajectory, trees

    return trajectory
