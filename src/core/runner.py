
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
import copy

from log_code.gen_renderings import save_gif_imageio

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
    azdetection=False, # If True, the agent will use the unrolling to detect problems
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
    
    if azdetection: # Calls the search function of azdetection

        tree = solver.search(env,planning_budget, observation, 0.0, original_env = original_env, n=unroll_steps)
        
    else: # Calls the standard search function

        tree = solver.search(env,planning_budget, observation, 0.0)

    step = 0

    while step < max_steps:
        
        if azdetection and solver.value_search and isinstance(tree, list):

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

            tree.backup_visits = 1 # For mcts-t only
            solver.backup(tree, tree.value_evaluation)
                       

        root_value = tree.value_evaluation # Contains the value estimate of the root node computed by the planning step

        tree.reset_var_val() # The value and variance (mvc) estimates of the whole subtree are reset.

        policy_dist = tree_evaluation_policy.softmaxed_distribution(tree) # Evaluates the tree using the given evaluation policy (e.g., visitation counts)

        if return_trees:
            trees.append(tree)

        if not azdetection: # We sample the action from the eval policy distribution
        
            distribution = th.distributions.Categorical(probs=custom_softmax(policy_dist.probs, temperature, None)) # apply extra softmax

            #print(distribution.probs)   
            action = distribution.sample().item() # Note that if the temperature of the softmax was zero, this becomes an argmax

        else: # If we are using azdetection, we need to check if a problem was detected

            if solver.problem_idx is None: # If no problem was detected, we act following the prior (quick)
                print("No problem detected, acting normally.")
                action = th.argmax(tree.prior_policy).item()

            else: # If a problem was detected, we act following the policy distribution

                distribution = th.distributions.Categorical(probs=custom_softmax(policy_dist.probs, temperature, None)) # apply extra softmax

                # Check if the probs are all equal, if so, we act according to the prior
                if th.all(th.eq(distribution.probs, distribution.probs[0])):

                # Check if more than 1 prob is non-zero
                #if th.sum(th.where(distribution.probs > 0, 1, 0)) > 1:
                    
                    print("All probs are equal, acting according to the prior.")
                    action = th.argmax(tree.prior_policy).item()
                    
                else:
                    action = distribution.sample().item() # Note that if the temperature of the softmax was zero, this becomes an argmax

            print(f"Env: action = {actions_dict[action]}")

        new_obs, reward, terminated, truncated, _ = env.step(action)

        if render:
            vis_env.step(action)
            frames.append(vis_env.render())

        # Convert the observation to a 2D position, hardcoded size of the grid for now
        new_pos_row = new_obs // 8 
        new_pos_col = new_obs % 8

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
        
        if azdetection:
            tree = solver.search(env, planning_budget, new_obs, reward, original_env=original_env, n=unroll_steps) 

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
    
    if render:
        save_gif_imageio(frames, output_path=f"gifs/output.gif", fps=5)

    return trajectory
