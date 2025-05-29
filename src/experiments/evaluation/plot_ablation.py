import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_ablation_subplots(filepaths_8x8, filepaths_16x16, labels_styles_colors, output_filename):
    """
    Plots ablation studies for 8x8 and 16x16 narrow configurations as subplots.

    Args:
        filepaths_8x8 (list): List of file paths for the 8x8 ablation study.
        filepaths_16x16 (list): List of file paths for the 16x16 ablation study.
        labels_styles_colors (list of tuples): List of tuples (label, color, linestyle, marker) for the plot.
        output_filename (str): Name of the output file to save the plot.
    """
    metric = "Discounted Return"
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot for 8x8
    optimal_value_8x8 = 0.95 ** ((8 * 2 - 2))
    for filepath, (label, color, linestyle, marker) in zip(filepaths_8x8, labels_styles_colors):
        df = pd.read_csv(filepath)
        axes[0].plot(df["Budget"], df[f"{metric} mean"], marker=marker, linestyle=linestyle, color=color, label=label)
        axes[0].fill_between(df["Budget"],
                             df[f"{metric} mean"] - df[f"{metric} SE"],
                             df[f"{metric} mean"] + df[f"{metric} SE"],
                             alpha=0.1, color=color)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(df["Budget"])
    axes[0].set_xticklabels(df["Budget"], fontsize=12)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[0].axhline(optimal_value_8x8, color='red', linestyle='dotted', linewidth=1.5, label="Optimal Value")
    axes[0].set_title("NARROW 8x8", fontsize=14)
    axes[0].set_xlabel("Planning Budget (log scale)", fontsize=12)
    axes[0].set_ylabel(metric, fontsize=12)
    axes[0].grid(True, linestyle="--", linewidth=0.5)
    axes[0].set_ylim(0, optimal_value_8x8 * 1.1)  # Set y-axis limit for 8x8

    # Plot for 16x16
    optimal_value_16x16 = 0.95 ** ((16 * 2 - 2))
    for filepath, (label, color, linestyle, marker) in zip(filepaths_16x16, labels_styles_colors):
        df = pd.read_csv(filepath)
        axes[1].plot(df["Budget"], df[f"{metric} mean"], marker=marker, linestyle=linestyle, color=color, label=label)
        axes[1].fill_between(df["Budget"],
                             df[f"{metric} mean"] - df[f"{metric} SE"],
                             df[f"{metric} mean"] + df[f"{metric} SE"],
                             alpha=0.1, color=color)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(df["Budget"])
    axes[1].set_xticklabels(df["Budget"], fontsize=12)
    axes[1].tick_params(axis='y', labelsize=12)
    axes[1].axhline(optimal_value_16x16, color='red', linestyle='dotted', linewidth=1.5, label="Optimal Value")
    axes[1].set_title("NARROW 16x16", fontsize=14)
    axes[1].set_xlabel("Planning Budget (log scale)", fontsize=12)
    axes[1].grid(True, linestyle="--", linewidth=0.5)
    axes[1].set_ylim(0, optimal_value_16x16 * 1.1)  # Set y-axis limit for 16x16

    # Collect unique legend entries from both axes
    handles_0, labels_0 = axes[0].get_legend_handles_labels()
    handles_1, labels_1 = axes[1].get_legend_handles_labels()

    # Combine and remove duplicates while preserving order
    from collections import OrderedDict
    combined = list(zip(handles_0 + handles_1, labels_0 + labels_1))
    unique = list(OrderedDict((label, handle) for handle, label in combined).items())

    # Unzip
    labels_unique, handles_unique = zip(*unique)

    fig.legend(handles_unique, labels_unique, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels_unique), fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(output_filename)
    plt.show()


if __name__ == "__main__":
    # Define labels, colors, linestyles, and markers
    labels_styles_colors = [
        ("PDDP", "purple", "-", "o"),
        ("C=1", "#D55E00", "-", "s"),
        ("NO TREE-REUSE", "#56B4E9", "-", "^"),
        ("NO DETPOL", "green", "-", "D"),
        ("NO VP", "gray", "-", "X"),
    ]

    # labels_styles_colors = [
    #     ("EDP", "black", "-", "o"),
    #     ("C=1", "#D55E00", "-", "s"),
    #     ("NO TREE-REUSE", "#56B4E9", "-", "^"),
    #     ("NO BLOCK LOOPS", "#E69F00", "-", "D"),
    # ]

    # # Filepaths for 8x8 narrow ablation
    filepaths_8x8 = [
        "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_NARROW.csv",
        "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(1.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_NARROW.csv",
        "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_False_8x8_NO_HOLES_NARROW.csv",
        "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(None)_Value_Penalty_1.0_treuse_True_8x8_NO_HOLES_NARROW.csv",
        "thesis_exp/8x8/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(10.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_0.0_treuse_True_8x8_NO_HOLES_NARROW.csv"
    ]

    # # Filepaths for 16x16 narrow ablation
    filepaths_16x16 = [
        "thesis_exp/16x16/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_NARROW.csv",
        "thesis_exp/16x16/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(1.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_NARROW.csv",
        "thesis_exp/16x16/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_1.0_treuse_False_16x16_NO_HOLES_NARROW.csv",
        "thesis_exp/16x16/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(None)_Value_Penalty_1.0_treuse_True_16x16_NO_HOLES_NARROW.csv",
        "thesis_exp/16x16/Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Beta_(1.0)_Predictor_(current_value)_eps_(0.05)_subthresh_(0.1)_ValueEst_(nn)_ttemp_(0.0)_Value_Penalty_0.0_treuse_True_16x16_NO_HOLES_NARROW.csv"
    ]

    # Filepaths for 8x8 slalom ablation
    # filepaths_8x8 = [
    #     "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #     "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(1.0)_Beta_(10.0)_treuse_True_bloops_True_ttemp_(None)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #     "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_treuse_False_bloops_True_ttemp_(None)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #     "thesis_exp/8x8/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_treuse_True_bloops_False_ttemp_(None)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    # ]

    # # Filepaths for 16x16 slalom ablation
    # filepaths_16x16 = [
    #     "thesis_exp/16x16/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_ttemp_(None)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #     "thesis_exp/16x16/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(1.0)_Beta_(10.0)_treuse_True_bloops_True_ttemp_(None)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #     "thesis_exp/16x16/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_treuse_False_bloops_True_ttemp_(None)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    #     "thesis_exp/16x16/Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.0)_Beta_(10.0)_treuse_True_bloops_False_ttemp_(None)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
    # ]

    # Generate the subplot
    plot_ablation_subplots(filepaths_8x8, filepaths_16x16, labels_styles_colors, "ablation_subplots.png")