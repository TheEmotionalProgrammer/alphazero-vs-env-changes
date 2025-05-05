import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

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

        optimal_value = 0.95 ** ((map_size * 2 - 2)) if config_label != "SLALOM" else 0.95 ** ((map_size * 2 - 2 + 8))

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
        ax.set_title(f"{config_label if config_label != 'DEFAULT' else 'SPARSE'}", fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.5)

        if ax == axes[0]:
            ax.set_ylabel(metric, fontsize=14)

    fig.supxlabel("Planning Budget (log scale)", fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=len(labels), fontsize=12)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.12, wspace=0.03)
    plt.savefig("comparison_subplots_customized.png")
    plt.show()


if __name__ == "__main__":

    map_size = 16

    # Define labels, colors, and line styles (shared across all subplots)
    labels_styles_colors = [
        ("AZ+PUCT", "blue", "-"),
        ("AZ+UCT", "blue", "--"),
        ("MINITREES", "red", "-"),
        ("MEGATREE", "orange", "-"),
    ]

    # FOR 8X8 FIRST EXPERIMENT
    # filepaths_dict = {
    #     "DEFAULT": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_DEFAULT.csv",
    #         "thesis_exp/8x8/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_DEFAULT.csv"
    #     ],
    #     "NARROW": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_NARROW.csv",
    #         "thesis_exp/8x8/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_NARROW.csv"
    #     ],
    #     "SLALOM": [
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_SLALOM.csv",
    #         "thesis_exp/8x8/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_n_(4)_eps_(0.08)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_8x8_NO_HOLES_SLALOM.csv"
    #     ],
    # }

    # FOR 16X16 FIRST EXPERIMENT
    filepaths_dict = {
        "DEFAULT": [
            "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
            "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_DEFAULT.csv",
            "thesis_exp/16x16/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_DEFAULT.csv",
            "thesis_exp/16x16/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_DEFAULT.csv"
        ],
        "NARROW": [
            "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
            "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_NARROW.csv",
            "thesis_exp/16x16/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_NARROW.csv",
            "thesis_exp/16x16/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_NARROW.csv"
        ],
        "SLALOM": [
            "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
            "thesis_exp/16x16/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_16x16_NO_HOLES_SLALOM.csv",
            "thesis_exp/16x16/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_SLALOM.csv",
            "thesis_exp/16x16/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_16x16_NO_HOLES_SLALOM.csv"
        ],
    }

    plot_comparison_subplots(filepaths_dict, labels_styles_colors, map_size)