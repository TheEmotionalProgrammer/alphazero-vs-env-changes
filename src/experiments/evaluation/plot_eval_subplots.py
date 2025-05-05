import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_comparison_subplots(filepaths_dict, labels=None, map_size=16, max_episode_length=100):
    """
    Plots 3 subplots comparing the same metrics across different CONFIG settings.

    Args:
        filepaths_dict (dict): Dictionary where keys are CONFIG labels, and values are lists of file paths.
        labels (list of str, optional): Labels for each file in the filepaths list.
    """

    metric = "Discounted Return"
    

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

    # Define a colormap and generate colors for the labels
    colormap = cm.rainbow
    color_indices = np.linspace(0, 1, 6)  # 6 colors for 6 labels
    colors = [colormap(idx) for idx in color_indices]

    for ax, (config_label, filepaths) in zip(axes, filepaths_dict.items()):

        optimal_value = 0.95 ** ((map_size * 2 - 2)) if config_label != "SLALOM" else 0.95 ** ((map_size * 2 - 2 + 4))

        dataframes = [pd.read_csv(filepath) for filepath in filepaths]

        for df, label in zip(dataframes, labels):

            # if "c=0.0" in label:
            #     linestyle = "-"
            #     color = colors[0]
            # elif "c=0.1" in label:
            #     linestyle = "-"
            #     color = colors[1]
            # elif "c=0.5" in label:
            #     linestyle = "-"
            #     color = colors[2]
            # elif "c=1.0" in label:
            #     linestyle = "-"
            #     color = colors[3]
            # elif "c=2.0" in label:
            #     linestyle = "-"
            #     color = colors[4]
            # elif "c=100.0" in label:
            #     linestyle = "-"
            #     color = colors[5]
            # else:
            #     linestyle = "-"
            #     color = None

            # if "c=0.0" in label and "+UCT" in label:
            #     linestyle = "-"
            #     color = colors[0]
            # elif "c=0.1" in label and "+UCT" in label:
            #     linestyle = "-"
            #     color = colors[1]
            # elif "c=0.0" in label and "PUCT" in label:
            #     linestyle = "-"
            #     color = colors[2]
            # elif "c=0.1" in label and "PUCT" in label:
            #     linestyle = "-"
            #     color = colors[3]

            if "Beta=1.0" in label and "+UCT" in label:
                linestyle = "-"
                color = colors[0]
            elif "Beta=10.0" in label and "+UCT" in label:
                linestyle = "-"
                color = colors[1]
            elif "Beta=1.0" in label and "PUCT" in label:
                linestyle = "-"
                color = colors[2]
            elif "Beta=10.0" in label and "PUCT" in label:
                linestyle = "-"
                color = colors[3]


            ax.plot(df["Budget"], df[f"{metric} mean"], marker="o", linestyle=linestyle, color=color, label=label)

            ax.fill_between(df["Budget"],
                            df[f"{metric} mean"] - df[f"{metric} SE"],
                            df[f"{metric} mean"] + df[f"{metric} SE"],
                            alpha=0.1, color=color)

        ax.set_xscale("log", base=2)
        ax.set_xticks(dataframes[0]["Budget"])
        ax.set_xticklabels(dataframes[0]["Budget"], fontsize=12)
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
    plt.savefig("comparison_subplots.png")
    plt.show()


if __name__ == "__main__":

    map_size = 16

    TRAIN_CONFIG = "NO_HOLES"
    #TRAIN_CONFIG = "MAZE_LR"
    #TRAIN_CONFIG = "MAZE_RL"

    vfunc = "nn"
    SELPOL = "UCT"

    CONFIGS = ["DEFAULT", "NARROW", "SLALOM"]  
    #CONFIGS = ["MAZE_LL", "MAZE_RR", "MAZE_LR"]
    #CONFIGS = ["MAZE_LL", "MAZE_RR", "MAZE_LR"]
    

    #labels = [f"AZ+{SELPOL} c=0.0", f"AZ+{SELPOL} c=0.1", f"AZ+{SELPOL} c=0.5", f"AZ+{SELPOL} c=1.0", f"AZ+{SELPOL} c=2.0", f"AZ+{SELPOL} c=100.0"]
    #labels = ["MINITREES+UCT c=0.0", "MINITREES+UCT c=0.1",  "MINITREES+PUCT c=0.0", "MINITREES+PUCT c=0.1"]
    labels = ["MEGA+UCT c=0.1 Beta=1.0", "MEGA+UCT c=0.1 Beta=10.0", "MEGA+PUCT c=0.1 Beta=1.0", "MEGA+PUCT c=0.1 Beta=10.0"]

    # filepaths_dict = {
    #     cfg: [
    #         f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({SELPOL})_c_(0.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #         f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({SELPOL})_c_(0.1)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #         f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({SELPOL})_c_(0.5)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #         f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({SELPOL})_c_(1.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #         f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({SELPOL})_c_(2.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #         f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({SELPOL})_c_(100.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #     ] for cfg in CONFIGS
    # }

    # filepaths_dict = {
    #     cfg: [
    #         f"thesis_exp/{map_size}x{map_size}/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #         f"thesis_exp/{map_size}x{map_size}/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #         f"thesis_exp/{map_size}x{map_size}/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #         f"thesis_exp/{map_size}x{map_size}/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #     ] for cfg in CONFIGS
    # }


    filepaths_dict = {
        cfg: [
            f"thesis_exp/{map_size}x{map_size}/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
            f"thesis_exp/{map_size}x{map_size}/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
            f"thesis_exp/{map_size}x{map_size}/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
            f"thesis_exp/{map_size}x{map_size}/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_(nn)_UpdateEst_(True)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
        ] for cfg in CONFIGS
    }

    plot_comparison_subplots(filepaths_dict, labels, map_size)
