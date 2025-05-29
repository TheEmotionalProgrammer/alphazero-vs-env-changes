import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

map_size = 8

def dict_configs (map_size):
    return {
        "DEFAULT": f"SPARSE {map_size}x{map_size}",
        "NARROW": f"NARROW {map_size}x{map_size}",
        "SLALOM": f"SLALOM {map_size}x{map_size}",
        "MAZE_LR": "MAZE_LR",
        "MAZE_RL": "MAZE_RL",
        "MAZE_LL": "MAZE_LL",
        "MAZE_RR": "MAZE_RR",
    }

def plot_comparison_subplots(filepaths_dict, labels_styles_colors, map_size=16, max_episode_length=100):
    """
    Plots 3 subplots comparing the same metrics across different CONFIG settings.

    Args:
        filepaths_dict (dict): Dictionary where keys are CONFIG labels, and values are lists of file paths.
        labels_styles_colors (list of tuples): List of tuples (label, color, linestyle) for all subplots.
    """

    metric = "Discounted Return"

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

    for ax, (config_label, filepaths) in zip(axes, filepaths_dict.items()):

        optimal_value = 0.95 ** ((map_size * 2 - 2)) if config_label != "SLALOM" else 0.95 ** ((map_size * 2 - 2 + (4 if map_size == 8 else 6)))
        #optimal_value = 0.95 ** ((map_size * 2 - 2)) if config_label != "MAZE_RL" else 0.95 ** 20

        for filepath, (label, color, linestyle) in zip(filepaths, labels_styles_colors):
            df = pd.read_csv(filepath)

            ax.plot(df["Budget"], df[f"{metric} mean"], marker="o", linestyle=linestyle, color=color, label=label)

            ax.fill_between(df["Budget"],
                            df[f"{metric} mean"] - df[f"{metric} SE"],
                            df[f"{metric} mean"] + df[f"{metric} SE"],
                            alpha=0.1, color=color)

        ax.set_xscale("log", base=2)
        ax.set_xticks(df["Budget"])
        ax.set_xticklabels(df["Budget"], fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.axhline(optimal_value, color='red', linestyle='dotted', linewidth=1.5, label="Optimal")
        ax.set_title(dict_configs[config_label], fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.5)

        if ax == axes[0]:
            ax.set_ylabel(metric, fontsize=14)

    fig.supxlabel("Planning Budget (log scale)", fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=len(labels), fontsize=12)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.12, wspace=0.03)
    plt.savefig("comparison_subplots_customized.png")
    plt.show()

def plot_combined_comparison(filepaths_dict_cw, filepaths_dict_ccw, labels_styles_colors, map_size=16, output_filename="combined_comparison.png"):
    """
    Plots a combined visualization with CW configurations on top and CCW configurations below.

    Args:
        filepaths_dict_cw (dict): Dictionary of file paths for CW configurations.
        filepaths_dict_ccw (dict): Dictionary of file paths for CCW configurations.
        labels_styles_colors (list of tuples): List of tuples (label, color, linestyle) for all subplots.
        map_size (int): Size of the map (e.g., 8 or 16).
        output_filename (str): Name of the output file to save the plot.
    """
    metric = "Discounted Return"
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)

    # Plot CW configurations (top row)
    for ax, (config_label, filepaths) in zip(axes[0], filepaths_dict_cw.items()):
        optimal_value = 0.95 ** ((map_size * 2 - 2)) if config_label != "SLALOM" else 0.95 ** ((map_size * 2 - 2 + (4 if map_size == 8 else 6)))
        for filepath, (label, color, linestyle) in zip(filepaths, labels_styles_colors):
            df = pd.read_csv(filepath)
            ax.plot(df["Budget"], df[f"{metric} mean"], marker="o", linestyle=linestyle, color=color, label=label)
            ax.fill_between(df["Budget"],
                            df[f"{metric} mean"] - df[f"{metric} SE"],
                            df[f"{metric} mean"] + df[f"{metric} SE"],
                            alpha=0.1, color=color)
        ax.set_xscale("log", base=2)
        ax.set_xticks(df["Budget"])
        ax.set_xticklabels(df["Budget"], fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.axhline(optimal_value, color='red', linestyle='dotted', linewidth=1.5, label="Optimal")
        ax.set_title(f"CW {dict_configs[config_label]}", fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.5)
        if ax == axes[0, 0]:
            ax.set_ylabel(metric, fontsize=14)

    # Plot CCW configurations (bottom row)
    for ax, (config_label, filepaths) in zip(axes[1], filepaths_dict_ccw.items()):
        optimal_value = 0.95 ** ((map_size * 2 - 2)) if config_label != "SLALOM" else 0.95 ** ((map_size * 2 - 2 + 4))
        for filepath, (label, color, linestyle) in zip(filepaths, labels_styles_colors):
            df = pd.read_csv(filepath)
            ax.plot(df["Budget"], df[f"{metric} mean"], marker="o", linestyle=linestyle, color=color, label=label)
            ax.fill_between(df["Budget"],
                            df[f"{metric} mean"] - df[f"{metric} SE"],
                            df[f"{metric} mean"] + df[f"{metric} SE"],
                            alpha=0.1, color=color)
        ax.set_xscale("log", base=2)
        ax.set_xticks(df["Budget"])
        ax.set_xticklabels(df["Budget"], fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.axhline(optimal_value, color='red', linestyle='dotted', linewidth=1.5, label="Optimal")
        ax.set_title(f"CCW {dict_configs[config_label]}", fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.5)
        if ax == axes[1, 0]:
            ax.set_ylabel(metric, fontsize=14)

    # Add shared x-axis label
    fig.supxlabel("Planning Budget (log scale)", fontsize=14)

    # Add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels), fontsize=12)

    plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.1, wspace=0.1, hspace=0.3)
    plt.savefig(output_filename)
    plt.show()

def plot_combined_mazes(filepaths_dict_maze_lr, filepaths_dict_maze_rl, labels_styles_colors, map_size=16, output_filename="combined_comparison.png"):
    """
    Plots a combined visualization with CW configurations on top and CCW configurations below.

    Args:
        filepaths_dict_cw (dict): Dictionary of file paths for CW configurations.
        filepaths_dict_ccw (dict): Dictionary of file paths for CCW configurations.
        labels_styles_colors (list of tuples): List of tuples (label, color, linestyle) for all subplots.
        map_size (int): Size of the map (e.g., 8 or 16).
        output_filename (str): Name of the output file to save the plot.
    """
    metric = "Discounted Return"
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    #{"↑" "→" "↓" "←"}
    # Plot MAZE_LR configurations (top row)
    for ax, (config_label, filepaths) in zip(axes[0], filepaths_dict_maze_lr.items()):
        optimal_value = 0.95 ** ((map_size * 2 - 2)) if config_label != "MAZE_RL" else 0.95 ** 20
        for filepath, (label, color, linestyle) in zip(filepaths, labels_styles_colors):
            df = pd.read_csv(filepath)
            ax.plot(df["Budget"], df[f"{metric} mean"], marker="o", linestyle=linestyle, color=color, label=label)
            ax.fill_between(df["Budget"],
                            df[f"{metric} mean"] - df[f"{metric} SE"],
                            df[f"{metric} mean"] + df[f"{metric} SE"],
                            alpha=0.1, color=color)
        ax.set_xscale("log", base=2)
        ax.set_xticks(df["Budget"])
        ax.set_xticklabels(df["Budget"], fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.axhline(optimal_value, color='red', linestyle='dotted', linewidth=1.5, label="Optimal")
        ax.set_title(f"MAZE_LR → {dict_configs(8)[config_label]}", fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.5)
        if ax == axes[0, 0]:
            ax.set_ylabel(metric, fontsize=14)

    # Plot MAZE_RL configurations (bottom row)
    for ax, (config_label, filepaths) in zip(axes[1], filepaths_dict_maze_rl.items()):
        optimal_value = 0.95 ** ((map_size * 2 - 2)) if config_label != "MAZE_RL" else 0.95 ** 20
        for filepath, (label, color, linestyle) in zip(filepaths, labels_styles_colors):
            df = pd.read_csv(filepath)
            ax.plot(df["Budget"], df[f"{metric} mean"], marker="o", linestyle=linestyle, color=color, label=label)
            ax.fill_between(df["Budget"],
                            df[f"{metric} mean"] - df[f"{metric} SE"],
                            df[f"{metric} mean"] + df[f"{metric} SE"],
                            alpha=0.1, color=color)
        ax.set_xscale("log", base=2)
        ax.set_xticks(df["Budget"])
        ax.set_xticklabels(df["Budget"], fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.axhline(optimal_value, color='red', linestyle='dotted', linewidth=1.5, label="Optimal")
        ax.set_title(f"MAZE_RL → {dict_configs(8)[config_label]}", fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.5)
        if ax == axes[1, 0]:
            ax.set_ylabel(metric, fontsize=14)

    # Add shared x-axis label
    fig.supxlabel("Planning Budget (log scale)", fontsize=14)

    # Add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels), fontsize=12)

    plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.1, wspace=0.1, hspace=0.3)
    plt.savefig(output_filename)
    plt.show()

def plot_combined_8x8_16x16(filepaths_dict_8x8, filepaths_dict_16x16, labels_styles_colors, output_filename="combined_8x8_16x16.png"):
    """
    Creates a combined plot with 8x8 configurations on top and 16x16 configurations at the bottom,
    with row-wise shared y-axes.

    Args:
        filepaths_dict_8x8 (dict): Dictionary of file paths for 8x8 configurations.
        filepaths_dict_16x16 (dict): Dictionary of file paths for 16x16 configurations.
        labels_styles_colors (list of tuples): List of tuples (label, color, linestyle) for all subplots.
        output_filename (str): Name of the output file to save the plot.
    """
    metric = "Discounted Return"

    # Create subplots for 8x8 (top row) and 16x16 (bottom row) with row-wise shared y-axes
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey='row')

    # Plot 8x8 configurations (top row)
    for ax, (config_label, filepaths) in zip(axes[0], filepaths_dict_8x8.items()):
        optimal_value = 0.95 ** ((8 * 2 - 2)) if config_label != "SLALOM" else 0.95 ** ((8 * 2 - 2 + 4))
        for filepath, (label, color, linestyle) in zip(filepaths, labels_styles_colors):
            df = pd.read_csv(filepath)
            ax.plot(df["Budget"], df[f"{metric} mean"], marker="o", linestyle=linestyle, color=color, label=label)
            ax.fill_between(df["Budget"],
                            df[f"{metric} mean"] - df[f"{metric} SE"],
                            df[f"{metric} mean"] + df[f"{metric} SE"],
                            alpha=0.1, color=color)
        ax.set_xscale("log", base=2)
        ax.set_xticks(df["Budget"])
        ax.set_xticklabels(df["Budget"], fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.axhline(optimal_value, color='red', linestyle='dotted', linewidth=1.5, label="Optimal")
        ax.set_title(f"{dict_configs(8)[config_label]}", fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.5)
        if ax == axes[0, 0]:
            ax.set_ylabel(metric, fontsize=14)

    # Plot 16x16 configurations (bottom row)
    for ax, (config_label, filepaths) in zip(axes[1], filepaths_dict_16x16.items()):
        optimal_value = 0.95 ** ((16 * 2 - 2)) if config_label != "SLALOM" else 0.95 ** ((16 * 2 - 2 + 6))
        for filepath, (label, color, linestyle) in zip(filepaths, labels_styles_colors):
            df = pd.read_csv(filepath)
            ax.plot(df["Budget"], df[f"{metric} mean"], marker="o", linestyle=linestyle, color=color, label=label)
            ax.fill_between(df["Budget"],
                            df[f"{metric} mean"] - df[f"{metric} SE"],
                            df[f"{metric} mean"] + df[f"{metric} SE"],
                            alpha=0.1, color=color)
        ax.set_xscale("log", base=2)
        ax.set_xticks(df["Budget"])
        ax.set_xticklabels(df["Budget"], fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.axhline(optimal_value, color='red', linestyle='dotted', linewidth=1.5, label="Optimal")
        ax.set_title(f"{dict_configs(16)[config_label]}", fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.5)
        if ax == axes[1, 0]:
            ax.set_ylabel(metric, fontsize=14)

    # Add shared x-axis label
    fig.supxlabel("Planning Budget (log scale)", fontsize=14)

    # Add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels), fontsize=12)

    plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.1, wspace=0.1, hspace=0.3)
    plt.savefig(output_filename)
    plt.show()

if __name__ == "__main__":

    # Define labels, colors, and line styles (shared across all subplots)
    labels_styles_colors = [
        # ("AZ+PUCT", "blue", "-"),
        # ("AZ+UCT", "blue", "--"),
        # ("MVC+PUCT", "green", "-"),
        # ("MVC+UCT", "green", "--"),
        # ("MINITREES", "red", "-"),
        # ("MEGATREE", "orange", "-"),
        ("PDDP", "purple", "-"),
        ("EDP", "black", "-"),
    ]

    # BUMP = "CW_" # "CCW_" or "CW_"

    # filepaths_dict_cw_8x8 = {
    #     "DEFAULT": [
    #         f"thesis_exp/8x8/CW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/8x8/CW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/8x8/CW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/8x8/CW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/8x8/CW_Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/8x8/CW_Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/8x8/CW_Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_DEFAULT.csv"
    #     ],
    #     "NARROW": [
    #         f"thesis_exp/8x8/CW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/8x8/CW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/8x8/CW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/8x8/CW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/8x8/CW_Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/8x8/CW_Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/8x8/CW_Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_NARROW.csv"
    #     ],
    #     "SLALOM": [
    #         f"thesis_exp/8x8/CW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/8x8/CW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/8x8/CW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/8x8/CW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/8x8/CW_Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/8x8/CW_Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/8x8/CW_Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_SLALOM.csv"
    #     ],
    # }


    # filepaths_dict_ccw_8x8 = {
    #     "DEFAULT": [
    #         f"thesis_exp/8x8/CCW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/8x8/CCW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/8x8/CCW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/8x8/CCW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/8x8/CCW_Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/8x8/CCW_Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/8x8/CCW_Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_DEFAULT.csv"
    #     ],
    #     "NARROW": [
    #         f"thesis_exp/8x8/CCW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/8x8/CCW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/8x8/CCW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/8x8/CCW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/8x8/CCW_Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/8x8/CCW_Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/8x8/CCW_Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_NARROW.csv"
    #     ],
    #     "SLALOM": [
    #         f"thesis_exp/8x8/CCW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/8x8/CCW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/8x8/CCW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/8x8/CCW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/8x8/CCW_Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/8x8/CCW_Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/8x8/CCW_Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_SLALOM.csv"
    #     ],
    # }

    # filepaths_dict_cw_16x16 = {
    #     "DEFAULT": [
    #         f"thesis_exp/16x16/CW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/16x16/CW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/16x16/CW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/16x16/CW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/16x16/CW_Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/16x16/CW_Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/16x16/CW_Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_DEFAULT.csv"
    #     ],
    #     "NARROW": [
    #         f"thesis_exp/16x16/CW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/16x16/CW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/16x16/CW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/16x16/CW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/16x16/CW_Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/16x16/CW_Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/16x16/CW_Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_NARROW.csv"
    #     ],
    #     "SLALOM": [
    #         f"thesis_exp/16x16/CW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/16x16/CW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/16x16/CW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/16x16/CW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/16x16/CW_Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/16x16/CW_Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/16x16/CW_Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_SLALOM.csv"
    #     ],
    # }


    # filepaths_dict_ccw_16x16 = {
    #     "DEFAULT": [
    #         f"thesis_exp/16x16/CCW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/16x16/CCW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/16x16/CCW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/16x16/CCW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/16x16/CCW_Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/16x16/CCW_Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/16x16/CCW_Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_DEFAULT.csv"
    #     ],
    #     "NARROW": [
    #         f"thesis_exp/16x16/CCW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/16x16/CCW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/16x16/CCW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/16x16/CCW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/16x16/CCW_Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/16x16/CCW_Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/16x16/CCW_Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_NARROW.csv"
    #     ],
    #     "SLALOM": [
    #         f"thesis_exp/16x16/CCW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/16x16/CCW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/16x16/CCW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/16x16/CCW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/16x16/CCW_Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/16x16/CCW_Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/16x16/CCW_Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_SLALOM.csv"
    #     ],
    # }

    # filepaths_dict_8x8_azd = {
    #     "DEFAULT": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_DEFAULT.csv",
    #     ],
    #     "NARROW": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_NARROW.csv",
    #     ],
    #     "SLALOM": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_SLALOM.csv",
    #     ]
    # }

    # filepaths_dict_16x16_azd = {
    #     "DEFAULT": [
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_DEFAULT.csv",

    #     ],
    #     "NARROW": [
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_NARROW.csv",
    #     ],
    #     "SLALOM": [
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_SLALOM.csv",
    #     ]
    # }

    # filepaths_dict_8x8_complete = {
    #     "DEFAULT": [
    #         "thesis_exp/8x8/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv"
    #     ],
    #     "NARROW": [
    #         "thesis_exp/8x8/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv"
    #     ],
    #     "SLALOM": [
    #         "thesis_exp/8x8/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv"
    #     ]
    # }

    # filepaths_dict_16x16_complete = {
    #     "DEFAULT": [
    #         "thesis_exp/16x16/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv"

    #     ],
    #     "NARROW": [

    #         "thesis_exp/16x16/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv"
    #     ],
    #     "SLALOM": [
    #         "thesis_exp/16x16/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv"
    #     ]
    # }

    filepaths_maze_lr_pddp_edp = {
        "MAZE_LL": [
            "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(100.0)_Predictor_(current_value)_eps_(0.08)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_0.1_treuse_True_8x8_MAZE_LR_MAZE_LL.csv",
            "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_LL.csv"
        ],
        "MAZE_RR": [
            "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(100.0)_Predictor_(current_value)_eps_(0.08)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_0.1_treuse_True_8x8_MAZE_LR_MAZE_RR.csv",
            "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RR.csv"
        ],
        "MAZE_RL": [
            "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(100.0)_Predictor_(current_value)_eps_(0.08)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_0.1_treuse_True_8x8_MAZE_LR_MAZE_RL.csv",
            "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RL.csv"
        ],
    }

    filepaths_maze_rl_pddp_edp = {
        "MAZE_LL": [
            "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_0.3_treuse_True_8x8_MAZE_RL_MAZE_LL.csv",
            "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LL.csv"
        ],
        "MAZE_RR": [
            "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_0.3_treuse_True_8x8_MAZE_RL_MAZE_RR.csv",
            "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_RR.csv"
        ],
        "MAZE_LR": [
            "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_0.3_treuse_True_8x8_MAZE_RL_MAZE_LR.csv",
            "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LR.csv"
        ]
    }

    # filepaths_dict_8x8_pddp = {
    #     "DEFAULT": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_DEFAULT.csv"

    #     ],
    #     "NARROW": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_NARROW.csv"

    #     ],
    #     "SLALOM": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_SLALOM.csv"

    #     ]
    # }

    # filepaths_dict_16x16_pddp = {
    #     "DEFAULT": [
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_DEFAULT.csv"

    #     ],
    #     "NARROW": [
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_NARROW.csv"           
    #     ],
    #     "SLALOM": [
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_SLALOM.csv"            
    #     ]
    # }


    # filepaths_dict_8x8_edp = {
    #     "DEFAULT": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv"

    #     ],
    #     "NARROW": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv"

    #     ],
    #     "SLALOM": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #        "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv"

    #     ]
    # }

    # filepaths_dict_16x16_edp = {
    #     "DEFAULT": [
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv"

    #     ],
    #     "NARROW": [
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv"
    #     ],
    #     "SLALOM": [
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv"
    #     ]
    # }




    # FOR 8X8 FIRST EXPERIMENT
    # filepaths_dict = {
    #     "DEFAULT": [
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_DEFAULT.csv"
    #     ],
    #     "NARROW": [
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_NARROW.csv"
    #     ],
    #     "SLALOM": [
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/8x8/{BUMP}Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_SLALOM.csv"
    #     ],
    # }

    # FOR 16X16 FIRST EXPERIMENT
    # filepaths_dict = {
    #     "DEFAULT": [
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_DEFAULT.csv",
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_DEFAULT.csv"
    #     ],
    #     "NARROW": [
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_NARROW.csv",
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_NARROW.csv"
    #     ],
    #     "SLALOM": [
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_SLALOM.csv",
    #         f"thesis_exp/16x16/{BUMP}Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_SLALOM.csv"
    #     ],
    # }

    #FOR 8X8 PDDP VS BASELINES
    # filepaths_dict = {
    #     "DEFAULT": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_DEFAULT.csv"
    #     ],
    #     "NARROW": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_NARROW.csv"
    #     ],
    #     "SLALOM": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_SLALOM.csv"
    #     ],
    # }

    # FOR 16X16 PDDP VS BASELINES
    # filepaths_dict = {

    #     "DEFAULT": [
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_DEFAULT.csv"
    #     ],
    #     "NARROW": [
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_NARROW.csv"

    #     ],
    #     "SLALOM": [
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_SLALOM.csv"
    #     ],
    # }

    # # PDDP - FOR MAZE_LR TRAINING
    # filepaths_dict_maze_lr = {
    #     "MAZE_LL": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.0)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_LL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_LL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_LL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_LL.csv",
    #         "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(100.0)_Predictor_(current_value)_eps_(0.08)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_0.1_treuse_True_8x8_MAZE_LR_MAZE_LL.csv"
    #     ],
    #     "MAZE_RR": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.0)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RR.csv",
    #         "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(100.0)_Predictor_(current_value)_eps_(0.08)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_0.1_treuse_True_8x8_MAZE_LR_MAZE_RR.csv"
    #     ],
    #     "MAZE_RL": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.0)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RL.csv",
    #         "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(100.0)_Predictor_(current_value)_eps_(0.08)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_0.1_treuse_True_8x8_MAZE_LR_MAZE_RL.csv"
    #     ],
    # }

    # # PDDP - FOR MAZE_RL TRAINING
    # filepaths_dict_maze_rl = {
    #     "MAZE_LL": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LL.csv",
    #         "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_0.3_treuse_True_8x8_MAZE_RL_MAZE_LL.csv"
    #     ],
    #     "MAZE_RR": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_RR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_RR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_RR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_RR.csv",
    #         "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_0.3_treuse_True_8x8_MAZE_RL_MAZE_RR.csv"
    #     ],
    #     "MAZE_LR": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LR.csv",
    #         "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_0.3_treuse_True_8x8_MAZE_RL_MAZE_LR.csv"
    #     ],
    # }

    # EDP - FOR 8X8 FIRST EXPERIMENT
    # filepaths_dict = {
    #     "DEFAULT": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv"
    #     ],
    #     "NARROW": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv"
    #     ],
    #     "SLALOM": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv"

    #     ],
    # }

    # EDP - FOR 16X16 FIRST EXPERIMENT
    # filepaths_dict = {
    #     "DEFAULT": [
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv"
    #     ],
    #     "NARROW": [
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv"

    #     ],
    #     "SLALOM": [
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/16x16/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv"
    #     ],
    # }

    # EDP - FOR MAZE_LR TRAINING
    # filepaths_dict_maze_lr = {
    #     "MAZE_LL": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.0)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_LL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_LL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_LL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_LL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_LL.csv"
    #     ],
    #     "MAZE_RR": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.0)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RR.csv"

    #     ],
    #     "MAZE_RL": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.0)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RL.csv"
            
    #     ],
    # }

    # EDP - FOR MAZE_RL TRAINING
    # filepaths_dict_maze_rl = {
    #     "MAZE_LL": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LL.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LL.csv"
    #     ],
    #     "MAZE_RR": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_RR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_RR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_RR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_RR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_RR.csv"
    #     ],
    #     "MAZE_LR": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LR.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LR.csv"
    #     ],
    # }

    # PDDP ABLATION


    #plot_comparison_subplots(filepaths_dict, labels_styles_colors, map_size)
    #plot_combined_comparison(filepaths_dict_cw, filepaths_dict_ccw, labels_styles_colors, map_size=8)
    #plot_combined_8x8_16x16(filepaths_dict_8x8_complete, filepaths_dict_16x16_complete, labels_styles_colors)
    plot_combined_mazes(filepaths_maze_lr_pddp_edp, filepaths_maze_rl_pddp_edp, labels_styles_colors, map_size=8)
