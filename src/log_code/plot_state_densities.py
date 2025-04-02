import sys
sys.path.append('src/')
from core.mcts import Node
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def obs_to_cords(state, ncols):
    row, col = divmod(state, ncols)
    return row, col

def calculate_density(tree: Node, ncols, nrows):

    goal_coord = (ncols - 1, nrows - 1)

    visitation_counts = tree.state_visitation_counts()
    density = np.zeros((nrows, ncols))
    for state, count in visitation_counts.items():
        if state == None:
            row, col = goal_coord
        else:
            row, col = obs_to_cords(state, ncols)
        density[row, col] = count
    return density

def calculate_nn_value_means(tree: Node, ncols, nrows):
    
    goal_coord = (ncols - 1, nrows - 1)

    value_means = tree.mean_nn_values_dict()
    density = np.zeros((nrows, ncols))
    for state, value in value_means.items():
        if state == None:
            row, col = goal_coord
        else:
            row, col = obs_to_cords(state, ncols)
        density[row, col] = value
    return density

def calculate_policy_value_means(tree: Node, ncols, nrows):

    goal_coord = (ncols - 1, nrows - 1)

    policy_means = tree.mean_policy_values_dict()
    density = np.zeros((nrows, ncols))
    for state, value in policy_means.items():
        if state == None:
            row, col = goal_coord
        else:
            row, col = obs_to_cords(state, ncols)
        density[row, col] = value
    return density

def calculate_variance_means(tree: Node, ncols, nrows):
    
    goal_coord = (ncols - 1, nrows - 1)

    variance_means = tree.mean_variances_dict()
    density = np.zeros((nrows, ncols))
    for state, value in variance_means.items():
        if state == None:
            row, col = goal_coord
        else:
            row, col = obs_to_cords(state, ncols)
        density[row, col] = value
    return density

def plot_density(density, root_state, obst_coords, ncols, nrows, cmap, ax=None):

    goal_coord = (ncols - 1, nrows - 1)

    for (row, col) in obst_coords:
        density[row, col] = np.nan  # Remove numbers from the cliff cells

    # Mask the 0.0 entries by setting them to NaN
    density[density == 0.0] = np.nan

    if ax is None:
        fig, ax = plt.subplots()

    # Plot the heatmap on the specified axes
    sns.heatmap(
        density, 
        cmap=cmap, 
        cbar=False, 
        annot=True, 
        fmt='.2f', 
        mask=np.isnan(density), 
        center=0, 
        ax=ax, 
        linewidths=0.5,  # Add gridlines
        linecolor='black',  # Set gridline color
        annot_kws={"size": 10 if ncols == 8 else 6},  # Set font size for annotations
    )

    # Mark the root state by a black border
    root_row, root_col = obs_to_cords(root_state, ncols)
    ax.add_patch(plt.Rectangle((root_col, root_row), 1, 1, fill=False, color='green', lw=5, angle=0))
    ax.add_patch(plt.Rectangle(goal_coord, 1, 1, fill=False, color='goldenrod', lw=5, angle=0))

    for (row, col) in obst_coords:
        ax.add_patch(plt.Rectangle((col, row), 1, 1, fill=True, color='black', lw=0, angle=0))

    # Set ticks and labels
    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    # ax.set_xticklabels(range(ncols))
    # ax.set_yticklabels(range(nrows))
    ax.set_aspect('equal')  # Set aspect ratio to be equal, making each cell square

    # Add gridlines for better visibility
    ax.grid(visible=True, which='both', color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    pos = ax.get_position()

    fig.add_artist(plt.Rectangle((pos.x0, pos.y0), pos.width, pos.height, edgecolor='black', fill=False, lw=1))

    return ax

def plot_tree(tree: Node, obst_coords, ncols, nrows, cmap):
    density = calculate_density(tree, ncols, nrows)
    return plot_density(density, tree.observation, obst_coords, ncols, nrows, cmap)