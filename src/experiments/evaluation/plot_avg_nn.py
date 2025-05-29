import sys

sys.path.append("src/")

import gymnasium as gym

import numpy as np

import torch as th

from environments.register import register_all


from az.model import (
    AlphaZeroModel,
    models_dict
)


from environments.observation_embeddings import (
    CoordinateEmbedding,

)

from experiments.parameters import base_parameters, fz_env_descriptions, env_challenges

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

@th.no_grad()
def plot_avg_nn_values(models, desc):
    map_size = len(desc)
    desc_arr = np.array([list(row) for row in desc])
    obst_coords = set(map(tuple, np.argwhere(desc_arr == "H")))
    # Add terminal state coordinates
    terminal_state = (map_size-1, map_size-1)
    avg_values = np.zeros((map_size, map_size))
    mask = np.zeros((map_size, map_size), dtype=bool)

    for i in range(map_size):
        for j in range(map_size):
            if (i, j) in obst_coords or (i, j) == terminal_state:
                mask[i, j] = True
                continue
            disc_obs = i * map_size + j
            values = []
            for model in models:
                value = model.single_observation_forward(disc_obs)[0]
                values.append(value)
            avg_values[i, j] = np.mean(values)

    # Set obstacle cells to NaN for seaborn masking
    avg_values_masked = avg_values.copy()
    avg_values_masked[mask] = np.nan

    plt.figure(figsize=(6, 6))
    cmap = sns.color_palette("viridis", as_cmap=True)
    # Draw heatmap with annotations, mask obstacles
    ax = sns.heatmap(
        avg_values_masked,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        mask=mask,
        linewidths=0.5,
        linecolor='black',
        cbar=False,
        #cbar_kws={'label': 'Average NN Value'},
        square=True,
        annot_kws={"size": 12 if map_size == 8 else 8},  # Set font size for annotations   
    )
    # Set black for masked (obstacle) cells
    for (i, j), val in np.ndenumerate(mask):
        if val and (i, j) != terminal_state:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))
    # Terminal state cell
    ax.text(
        terminal_state[1] + 0.5, terminal_state[0] + 0.5, "$",
        fontsize=20 if map_size==8 else 12, ha='center', va='center', color='black'
    )

    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            map_size, map_size,
            fill=False,
            edgecolor='black',
            linewidth=0.5,
            clip_on=False
        )
    )

    # Explicitly set limits to perfectly match rectangle to heatmap grid
    ax.set_xlim(0, map_size)
    ax.set_ylim(map_size, 0)

    ax.set_xticks([])
    ax.set_yticks([])

    #plt.title('Average NN Values')
    #plt.xlabel('Column')
    #plt.ylabel('Row')
    plt.tight_layout()
    plt.show()

@th.no_grad()
def plot_std_nn_values(models, desc):
    map_size = len(desc)
    desc_arr = np.array([list(row) for row in desc])
    obst_coords = set(map(tuple, np.argwhere(desc_arr == "H")))
    # Add terminal state coordinates
    terminal_state = (map_size-1, map_size-1)
    # Initialize std_values and mask
    std_values = np.zeros((map_size, map_size))
    mask = np.zeros((map_size, map_size), dtype=bool)

    for i in range(map_size):
        for j in range(map_size):
            if (i, j) in obst_coords or (i, j) == terminal_state:
                mask[i, j] = True
                continue
            disc_obs = i * map_size + j
            values = []
            for model in models:
                value = model.single_observation_forward(disc_obs)[0]
                values.append(value)
            std_values[i, j] = np.std(values)

    # Set obstacle cells to NaN for seaborn masking
    std_values_masked = std_values.copy()
    std_values_masked[mask] = np.nan

    plt.figure(figsize=(6, 6))
    cmap = sns.color_palette("plasma", as_cmap=True)
    ax = sns.heatmap(
        std_values_masked,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        mask=mask,
        linewidths=0.5,
        linecolor='black',
        cbar=False,
        #cbar_kws={'label': 'NN Value Std. Dev.'},
        square=True,
        annot_kws={"size": 12 if map_size == 8 else 8},  # Set font size for annotations
    )
    # Set black for masked (obstacle) cells
    for (i, j), val in np.ndenumerate(mask):
        if val and (i, j) != terminal_state:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))
    # Terminal state cell
    ax.text(
        terminal_state[1] + 0.5, terminal_state[0] + 0.5, "$",
        fontsize=20 if map_size==8 else 12, ha='center', va='center', color='black'
    )

    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            map_size, map_size,
            fill=False,
            edgecolor='black',
            linewidth=0.5,
            clip_on=False
        )
    )

    # Explicitly set limits to perfectly match rectangle to heatmap grid
    ax.set_xlim(0, map_size)
    ax.set_ylim(map_size, 0)

    ax.set_xticks([])
    ax.set_yticks([])

    #plt.title('Standard Deviation of NN Values')
    # plt.xlabel('Column')
    # plt.ylabel('Row')
    plt.tight_layout()
    plt.show()

@th.no_grad()
def plot_avg_nn_policies(models, desc, title="Average NN Policy Logits"):
    map_size = len(desc)
    desc_arr = np.array([list(row) for row in desc])
    obst_coords = set(map(tuple, np.argwhere(desc_arr == "H")))
    # Add terminal state coordinates
    terminal_state = (map_size-1, map_size-1)
    avg_logits = np.zeros((map_size, map_size, 4))

    for i in range(map_size):
        for j in range(map_size):
            if (i, j) in obst_coords or (i, j) == terminal_state:
                avg_logits[i, j, :] = np.nan
                continue
            disc_obs = i * map_size + j
            logits = []
            for model in models:
                logit = model.single_observation_forward(disc_obs)[1]
                logits.append(logit)
            avg_logits[i, j, :] = np.mean(logits, axis=0)

    plt.figure(figsize=(6, 6))

    # Dummy seaborn heatmap to match exact format
    dummy_data = np.full((map_size, map_size), np.nan)
    dummy_data[0, 0] = 0  # Set single dummy value for white color
    ax = sns.heatmap(
        dummy_data,
        cmap=sns.color_palette("binary", as_cmap=True),
        cbar=False,
        linewidths=0.5,
        linecolor='black',
        square=True,
        mask=np.isnan(dummy_data)
    )

    # Obstacle cells
    for (i, j) in obst_coords:
        ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))
    # Terminal state cell
    #ax.add_patch(plt.Rectangle((terminal_state[1], terminal_state[0]), 1, 1, color='#FFD966'))
    # Add a dollar sign for the terminal state
    ax.text(
        terminal_state[1] + 0.5, terminal_state[0] + 0.5, "$",
        fontsize=20 if map_size==8 else 12, ha='center', va='center', color='black'
    )

    # Policy arrows setup
    directions = {
        0: (-1, 0),  # left
        1: (0, 1),   # down
        2: (1, 0),   # right
        3: (0, -1),  # up
    }
    colors = ["blue", "green", "orange", "red"]

    # Normalize and draw arrows
    for i in range(map_size):
        for j in range(map_size):
            if (i, j) in obst_coords or (i, j) == terminal_state:
                continue
            cell_logits = avg_logits[i, j, :]
            min_len, max_len = 0.4, 1.2
            if np.all(cell_logits == cell_logits[0]):
                arrow_lengths = np.full(4, min_len)
            else:
                norm = (cell_logits - np.min(cell_logits)) / (np.ptp(cell_logits) + 1e-8)
                arrow_lengths = min_len + norm * (max_len - min_len)

            for a in range(4):
                dx, dy = directions[a]
                length = arrow_lengths[a]
                scale = 0.25
                offset = 0.08
                start_x = j + dx * offset + 0.5
                start_y = i + dy * offset + 0.5
                ax.add_patch(
                    patches.FancyArrow(
                        start_x, start_y, dx * length * scale, dy * length * scale,
                        width=0.04, head_width=0.10, head_length=0.10,
                        length_includes_head=True, color=colors[a], alpha=0.8
                    )
                )

    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            map_size, map_size,
            fill=False,
            edgecolor='black',
            linewidth=0.5,
            clip_on=False
        )
    )

    # Explicitly set limits to perfectly match rectangle to heatmap grid
    ax.set_xlim(0, map_size)
    ax.set_ylim(map_size, 0)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_aspect('equal', adjustable='box')
    #plt.title(title)
    plt.tight_layout()
    plt.show()

@th.no_grad()
def plot_std_nn_policies(models, desc, title="Standard Deviation of NN Policy Logits"):
    map_size = len(desc)
    desc_arr = np.array([list(row) for row in desc])
    obst_coords = set(map(tuple, np.argwhere(desc_arr == "H")))
    terminal_state = (map_size-1, map_size-1)
    std_logits = np.zeros((map_size, map_size))
    mask = np.zeros((map_size, map_size), dtype=bool)

    for i in range(map_size):
        for j in range(map_size):
            if (i, j) in obst_coords or (i, j) == terminal_state:
                mask[i, j] = True
                std_logits[i, j] = np.nan
                continue
            disc_obs = i * map_size + j
            logits = []
            for model in models:
                logit = model.single_observation_forward(disc_obs)[1]
                logits.append(logit)
            logits = np.stack(logits, axis=0)  # shape: (num_models, 4)
            std_logits[i, j] = np.mean(np.std(logits, axis=0))  # mean std across actions

    # Set obstacle cells to NaN for seaborn masking
    std_logits_masked = std_logits.copy()
    std_logits_masked[mask] = np.nan

    plt.figure(figsize=(6, 6))
    cmap = sns.color_palette("plasma", as_cmap=True)
    ax = sns.heatmap(
        std_logits_masked,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        mask=mask,
        linewidths=0.5,
        linecolor='black',
        cbar=False,
        square=True,
        annot_kws={"size": 12 if map_size == 8 else 8},
    )
    for (i, j), val in np.ndenumerate(mask):
        if val and (i, j) != terminal_state:
            ax.add_patch(patches.Rectangle((j, i), 1, 1, color='black'))
    # Terminal state cell



    ax.text(
        terminal_state[1] + 0.5, terminal_state[0] + 0.5, "$",
        fontsize=20 if map_size==8 else 12, ha='center', va='center', color='black'
    )

    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            map_size, map_size,
            fill=False,
            edgecolor='black',
            linewidth=0.5,
            clip_on=False
        )
    )

    # Explicitly set limits to perfectly match rectangle to heatmap grid
    ax.set_xlim(0, map_size)
    ax.set_ylim(map_size, 0)

    ax.set_xticks([])
    ax.set_yticks([])
    #plt.title(title)
    # plt.xlabel('Column')
    # plt.ylabel('Row')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    
    # Example usage
    seeds = range(0, 10)

    map_size = 8

    hidden_dim = base_parameters["hidden_dim"]

    config = "MAZE_RL"  # "NO_HOLES", "MAZE_RL", "MAZE_LR"

    name = config if config != "NO_HOLES" else "NO_OBSTACLES"

    desc = fz_env_descriptions[f"{map_size}x{map_size}_{name}"]

    if map_size == 8 and config == "NO_HOLES":
        models = [f"hyper/AZTrain_env=CustomFrozenLakeNoHoles8x8-v1_evalpol=visit_iterations=50_budget=64_df=0.95_lr=0.001_nstepslr=2_seed={seed}/checkpoint.pth" for seed in seeds]
    elif map_size == 16 and config == "NO_HOLES":
        models = [f"hyper/AZTrain_env=CustomFrozenLakeNoHoles16x16-v1_evalpol=visit_iterations=60_budget=128_df=0.95_lr=0.003_nstepslr=2_seed={seed}/checkpoint.pth" for seed in seeds]

    elif map_size == 8 and config == "MAZE_RL":
        models = [f"hyper/AZTrain_env=8x8_MAZE_RL_evalpol=visit_iterations=150_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={seed}/checkpoint.pth" for seed in seeds]

    elif map_size == 8 and config == "MAZE_LR":
        models = [f"hyper/AZTrain_env=8x8_MAZE_LR_evalpol=visit_iterations=100_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={seed}/checkpoint.pth" for seed in seeds]

    challenge = env_challenges[f"CustomFrozenLakeNoHoles{map_size}x{map_size}-v1"]
    challenge["env_params"]["desc"] = desc
    challenge["env_params"]["is_slippery"] = False
    challenge["env_params"]["hole_reward"] = 0
    challenge["env_params"]["terminate_on_hole"] = False
    challenge["env_params"]["deviation_type"] = "bump"

    env = gym.make(**challenge["env_params"])

    models = [models_dict["seperated"].load_model(model, env, False, hidden_dim) for model in models]

    plot_avg_nn_values(models, desc)
    plot_avg_nn_policies(models, desc)
    plot_std_nn_values(models, desc)
    plot_std_nn_policies(models, desc)
    

    # def obs_to_tensor(self, observation, *args, **kwargs):

    #     """
    #     Returns a tensor of shape (2,) with the coordinates of the observation,
    #     scaled to the range [-1, 1].
    #     """

    #     cords = divmod(observation, self.ncols) 
    #     cords = (np.array(cords) / np.array([self.nrows-1, self.ncols-1])) * self.multiplier + self.shift

    #     return th.tensor(cords, *args, **kwargs)

