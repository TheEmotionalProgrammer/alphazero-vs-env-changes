import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

map_size = 8

dict_configs = {
        "DEFAULT": f"SPARSE {map_size}x{map_size}",
        "NARROW": f"NARROW {map_size}x{map_size}",
        "SLALOM": f"SLALOM {map_size}x{map_size}",
        "MAZE_LR": "MAZE_LR",
        "MAZE_RL": "MAZE_RL",
        "MAZE_LL": "MAZE_LL",
        "MAZE_RR": "MAZE_RR",
    }

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

        optimal_value = 0.95 ** ((map_size * 2 - 2)) if config_label != "SLALOM" else 0.95 ** ((map_size * 2 - 2 + (4 if map_size == 8 else 6)))

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

            # if "Beta=1.0" in label and "+UCT" in label:
            #     linestyle = "-"
            #     color = colors[0]
            # elif "Beta=10.0" in label and "+UCT" in label:
            #     linestyle = "-"
            #     color = colors[1]
            # elif "Beta=1.0" in label and "PUCT" in label:
            #     linestyle = "-"
            #     color = colors[2]
            # elif "Beta=10.0" in label and "PUCT" in label:
            #     color = colors[3]

            if "β=1.0" in label and "c=0.0" in label:
                linestyle = "-"
                color = colors[0]
            elif "β=10.0" in label and "c=0.0" in label:
                linestyle = "-"
                color = colors[2]
            elif "β=100.0" in label and "c=0.0" in label:
                linestyle = "-"
                color = colors[5]
            elif "β=1.0" in label and "c=0.1" in label:
                linestyle = "--"
                color = colors[0]
            elif "β=10.0" in label and "c=0.1" in label:
                linestyle = "--"
                color = colors[2]
            elif "β=100.0" in label and "c=0.1" in label:
                linestyle = "--"
                color = colors[5]


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
        ax.set_title(dict_configs[config_label], fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.5)

        if ax == axes[0]:
            ax.set_ylabel(metric, fontsize=14)


    fig.supxlabel("Planning Budget (log scale)", fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=len(labels), fontsize=12)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.12, wspace=0.03)
    plt.savefig("comparison_subplots.png")
    plt.show()


def plot_comparison_subplots_2x3(filepaths_dict, labels=None, map_size=16, max_episode_length=100, selpols=("PUCT", "UCT"), train_config=None):
    """
    Plots 2x3 subplots comparing the same metrics across different CONFIG settings and SELPOLs.

    Args:
        filepaths_dict (dict): Nested dict: {SELPOL: {CONFIG: [filepaths]}}
        labels (list of str, optional): Labels for each file in the filepaths list.
        selpols (tuple): Tuple of SELPOL names, e.g., ("PUCT", "UCT")
    """
    metric = "Discounted Return"
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)

    colormap = cm.rainbow
    color_indices = np.linspace(0, 1, 6)
    colors = [colormap(idx) for idx in color_indices]

    for row, selpol in enumerate(selpols):
        for col, (config_label, filepaths) in enumerate(filepaths_dict[selpol].items()):
            ax = axes[row, col]
            optimal_value = 0.95 ** ((map_size * 2 - 2)) if config_label != "SLALOM" else 0.95 ** ((map_size * 2 - 2 + (4 if map_size == 8 else 6)))
            if config_label == "MAZE_RL":
                optimal_value = 0.95 ** 20
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
                if "β=1.0" in label and "c=0.0" in label:
                    linestyle = "-"
                    color = colors[0]
                elif "β=10.0" in label and "c=0.0" in label:
                    linestyle = "-"
                    color = colors[2]
                elif "β=100.0" in label and "c=0.0" in label:
                    linestyle = "-"
                    color = colors[5]
                elif "β=1.0" in label and "c=0.1" in label:
                    linestyle = "--"
                    color = colors[0]
                elif "β=10.0" in label and "c=0.1" in label:
                    linestyle = "--"
                    color = colors[2]
                elif "β=100.0" in label and "c=0.1" in label:
                    linestyle = "--"
                    color = colors[5]

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
            if train_config is None:
                ax.set_title(f"{selpol} - {dict_configs[config_label]}", fontsize=14)
            else:
                ax.set_title(f"{selpol} - {train_config} → {dict_configs[config_label]}", fontsize=14)
                
            ax.grid(True, linestyle="--", linewidth=0.5)
            if col == 0:
                ax.set_ylabel(metric, fontsize=14)

    fig.supxlabel("Planning Budget (log scale)", fontsize=14)
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels_), fontsize=12)
    plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.1, wspace=0.1, hspace=0.3)
    plt.savefig("comparison_subplots_2x3.png")
    plt.show()

if __name__ == "__main__":

    TRAIN_CONFIG = "MAZE_RL"
    map_size = 8
    #CONFIGS = ["DEFAULT", "NARROW", "SLALOM"]
    CONFIGS = ["MAZE_LL", "MAZE_RR", "MAZE_LR"]
    selpols = ("PUCT", "UCT")

    labels = [
        f"c=0.0, β=1.0",
        f"c=0.0, β=10.0",
        f"c=0.0, β=100.0",
        f"c=0.1, β=1.0",
        f"c=0.1, β=10.0",
        f"c=0.1, β=100.0",
    ]
    #labels = [f"c=0.0", f"c=0.1", f"c=0.5", f"c=1.0", f"c=2.0", f"c=100.0"]

    # filepaths_dict = {
    #     selpol: {
    #         cfg: [
    #             f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(0.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #             f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(0.1)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #             f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(0.5)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #             f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(1.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #             f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(2.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #             f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(100.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #         ] for cfg in CONFIGS
    #     } for selpol in selpols
    # }

    filepaths_dict = {
        selpol: {
            cfg: [
                f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_({"PolicyUCT" if selpol=="UCT" else "PolicyPUCT"})_c_(0.0)_Beta_(1.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
                f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_({"PolicyUCT" if selpol=="UCT" else "PolicyPUCT"})_c_(0.0)_Beta_(10.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
                f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_({"PolicyUCT" if selpol=="UCT" else "PolicyPUCT"})_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
                f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_({"PolicyUCT" if selpol=="UCT" else "PolicyPUCT"})_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
                f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_({"PolicyUCT" if selpol=="UCT" else "PolicyPUCT"})_c_(0.1)_Beta_(10.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
                f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_({"PolicyUCT" if selpol=="UCT" else "PolicyPUCT"})_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
            ] for cfg in CONFIGS
        } for selpol in selpols
    }

    # Format labels for each SELPOL
    labels_puct = [lbl.format(SELPOL="PUCT") for lbl in labels]
    labels_uct = [lbl.format(SELPOL="UCT") for lbl in labels]

    # Call the new plotting function
    plot_comparison_subplots_2x3(filepaths_dict, labels_puct, map_size, selpols=selpols, train_config=TRAIN_CONFIG)
