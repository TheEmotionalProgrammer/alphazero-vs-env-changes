from ast import Not
from collections import Counter
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from az.model import AlphaZeroModel

from environments.observation_embeddings import CoordinateEmbedding, MiniGridEmbedding

@th.no_grad()
def investigate_model(model: AlphaZeroModel):
    """
    returns a dict of {obs: value} for each obs in the observation space
    """
    tensor_observations = []

    # Convert each observation into a tensor and add it to the list
    for obs in range(model.observation_embedding.observation_space.n):
        tensor_obs = model.observation_embedding.obs_to_tensor(obs, dtype=th.float32)
        tensor_observations.append(tensor_obs)

    # Stack all tensor observations into a single batch
    batch = th.stack(tensor_observations)

    # Pass the batch through the model
    values, policies = model(batch)


    return {obs: (value, policy) for obs, value, policy in zip(range(model.observation_embedding.observation_space.n), values, policies)}

def create_figure_and_axes():
    fig, ax = plt.subplots()
    ax.grid(False)
    ax.axis("off")
    return fig, ax

def plot_image(fig, ax, image, title):
    ax.imshow(image, interpolation="nearest")
    ax.set_title(title)
    plt.tight_layout()
    if fig is not None:
        plt.close(fig) 

import matplotlib.patches as patches

def plot_value_network(outputs, nrows, ncols, title="Value Network"):
    plt.ioff()
    grid = np.zeros((nrows, ncols))
    for state, value in outputs.items():
        row, col = divmod(state, ncols)
        grid[row, col] = value[0]
    
    fig, ax = plt.subplots()
    #ax.grid(False)

    for i in range(nrows):
        for j in range(ncols):
            # Add text in the cell
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="red", fontsize=7)
            # Add a rectangle around the cell for the border
            # rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=0.5, edgecolor='black', facecolor='none')
            # ax.add_patch(rect)

    ax.imshow(grid, interpolation="nearest")
    ax.set_title(title)
    
    # Adjust the grid to make it more clear
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)
    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig

import matplotlib.patches as patches

def plot_policy_network(outputs, nrows=4, ncols=12, title="Policy Network"):
    plt.ioff()
    action_arrows = {3: "↑", 2: "→", 1: "↓", 0: "←"}
    preferred_actions = np.zeros((nrows, ncols), dtype="<U2")
    
    # Fill the grid with preferred actions
    for state, action in outputs.items():
        row, col = divmod(state, ncols)
        preferred_actions[row, col] = action_arrows[np.argmax(action[1]).item()]
    
    fig, ax = plt.subplots()

    # Set the background color to white for all cells
    ax.imshow(np.ones((nrows, ncols)), interpolation="nearest", cmap="binary", vmin=1, vmax=1)
    
    # Add text and cell borders
    for i in range(nrows):
        for j in range(ncols):
            # Add preferred action text
            ax.text(j, i, f"{preferred_actions[i, j]}", ha="center", va="center", color="red", fontsize=14)
    
    ax.set_title(title)

    # Adjust the grid to make it more clear
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)
    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig



def plot_visits_with_counter(
    visit_counts: Counter,
    observation_embedding: CoordinateEmbedding | MiniGridEmbedding,
    step,
    title="State Visit Counts",
):  
    if isinstance(observation_embedding, CoordinateEmbedding):
        grid = np.zeros((observation_embedding.nrows, observation_embedding.ncols))

        # Populate the grid with visit counts
        for obs in range(observation_embedding.observation_space.n):
            obs_tensor = tuple(
                observation_embedding.obs_to_tensor(obs, dtype=th.float32).tolist()
            )
            count = visit_counts.get(obs_tensor, 0)
            row, col = divmod(obs, observation_embedding.ncols)
            grid[row, col] = count

        fig, ax = plt.subplots()
        ax.imshow(grid, cmap="viridis", interpolation="nearest")
        ax.set_title(f"{title}, Step: {step}")
        
        # Add text and cell borders
        for obs in range(observation_embedding.observation_space.n):
            row, col = divmod(obs, observation_embedding.ncols)
            # Add visit count as text
            ax.text(
                col, row, f"{grid[row, col]:.0f}", ha="center", va="center", color="red", fontsize=6
            )
            # Add a rectangle around the cell for the border
            rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1, linewidth=0.5, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
        
        # Adjust the grid for clarity
        ax.set_xlim(-0.5, observation_embedding.ncols - 0.5)
        ax.set_ylim(observation_embedding.nrows - 0.5, -0.5)
        ax.set_xticks(range(observation_embedding.ncols))
        ax.set_yticks(range(observation_embedding.nrows))
        ax.set_xticks(np.arange(-0.5, observation_embedding.ncols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, observation_embedding.nrows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        return fig
    
    elif isinstance(observation_embedding, MiniGridEmbedding):
        grid = np.zeros((observation_embedding.height, observation_embedding.width))

        # Iterate over all cells in the grid
        for row in range(observation_embedding.height):
            for col in range(observation_embedding.width):
                # Create a mock observation with direction as 0 (default)
                obs = np.zeros((observation_embedding.height, observation_embedding.width))
                obs[row, col] = 1  # Set the agent position
                obs_tensor = tuple(
                    observation_embedding.obs_to_tensor(obs, dtype=th.float32).tolist()
                )
                count = visit_counts.get(obs_tensor, 0)
                grid[row, col] = count

        fig, ax = plt.subplots()
        ax.imshow(grid, cmap="viridis", interpolation="nearest")
        ax.set_title(f"{title}, Step: {step}")
        for row in range(observation_embedding.height):
            for col in range(observation_embedding.width):
                ax.text(
                    col, row, f"{grid[row, col]:.0f}", ha="center", va="center", color="white"
                )
        plt.tight_layout()
        return fig
