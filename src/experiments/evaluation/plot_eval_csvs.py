import pandas as pd
import matplotlib.pyplot as plt

def plot_comparison_from_csvs(filepaths, labels=None, map_size=16, max_episode_length=100):
    """
    Load multiple CSVs and plot their metrics on the same graph for comparison,
    with min-max rescaling and reference lines.

    Args:
        filepaths (list of str): List of CSV file paths.
        labels (list of str, optional): List of labels for each CSV file (defaults to filenames).
    """
    if labels is None:
        labels = [f"Config {i+1}" for i in range(len(filepaths))]

    # Define min-max limits for each metric
    min_max_dict = {
        "Discounted Return": [0, 0.95**((map_size*2 - 2))],  # Min: 0, Max: Optimal value
        "Return": [0, 1],                    # Min: 0, Max: 1 (perfect return)
        "Episode Length": [0, 100],          # Min: 0, Max: Episode Length)
    }

    # Define optimal values for reference lines
    optimal_values = {
        "Discounted Return": 0.95**((map_size*2 - 2)),
        "Return": 1.0,  
        "Episode Length": 14  # No optimal value for episode length
    }

    # Function to adjust y-axis limits with padding
    def adjust_ylim(metric):
        min_val, max_val = min_max_dict[metric]
        range_padding = (max_val - min_val) * 0.05  # 5% of the range
        return min_val - range_padding, max_val + range_padding

    # Load data from each CSV and store it
    dataframes = [pd.read_csv(filepath) for filepath in filepaths]

    for metric in ["Discounted Return", "Return", "Episode Length"]:
        plt.figure(figsize=(8, 6))

        for df, label in zip(dataframes, labels):
            # Ensure we have correct columns
            if f"{metric} mean" not in df.columns or f"{metric} SE" not in df.columns:
                print(f"Skipping {label}: Missing required columns in {filepaths}")
                continue

            # Determine color and linestyle based on label
            if "AZ" in label:
                color = "blue"
            elif "MTVS" in label:
                color = "orange"
            elif "MVC" in label:
                color = "green"
            elif "CAP" in label or "STANDARD" in label:
                color = "purple"
            else:
                color = None


            if "PUCT" in label:
                linestyle = "-"
            elif "UCT" in label:
                linestyle = "--"
            else:
                linestyle = "-"

            # if "c=0.0" in label:
            #     linestyle = "-"
            #     color = "#001F3F"
            # elif "c=0.5" in label:
            #     linestyle = "-"
            #     color = "#191970"
            # elif "c=1.0" in label:
            #     linestyle = "-"
            #     color = "blue"
            # elif "c=2.0" in label:
            #     linestyle = "-"
            #     color = "#4169E1"
            # elif "c=100.0" in label:
            #     linestyle = "-"
            #     color = "#87CEEB"
            # else:
            #     linestyle = "-"

            # if "c=0.0" in label:
            #     linestyle = "-"
            #     color = "#006400"  # Dark Green
            # elif "c=0.5" in label:
            #     linestyle = "-"
            #     color = "green"  # Forest Green
            # elif "c=1.0" in label:
            #     linestyle = "-"
            #     color = "#00A86B"  # Default Green
            # elif "c=2.0" in label:
            #     linestyle = "-"
            #     color = "#32CD32"  # Lime Green
            # elif "c=100.0" in label:
            #     linestyle = "-"
            #     color = "#98FB98"  # Pale Green
            # else:
            #     linestyle = "-"

            # if "NO TREE REUSE" in label:
            #     linestyle = "-"
            #     color = "darkred"
            # elif "C>0" in label:
            #     linestyle = "-"
            #     color = "goldenrod"
            # elif "NO VP" in label:
            #     linestyle = "-"
            #     color = "orange"
            # elif "NONE TEMP" in label:
            #     linestyle = "-"
            #     color = "red"


            # Plot mean values
            plt.plot(df["Budget"], df[f"{metric} mean"], marker="o", linestyle=linestyle, color=color, label=label)

            # Fill shaded area for standard error
            plt.fill_between(df["Budget"], 
                             df[f"{metric} mean"] - df[f"{metric} SE"], 
                             df[f"{metric} mean"] + df[f"{metric} SE"], 
                             alpha=0.1, color=color)

        # Set x-axis to logarithmic scale
        plt.xscale("log", base=2)

        # Set x-ticks explicitly
        plt.xticks(dataframes[0]["Budget"], labels=dataframes[0]["Budget"], fontsize=12)
        plt.yticks(fontsize=12)

        # Adjust y-axis limits dynamically
        plt.ylim(adjust_ylim(metric))

        # Add a horizontal reference line for optimal values (if applicable)
        if metric in optimal_values:
            plt.axhline(optimal_values[metric], color='red', linestyle='dotted', linewidth=1.5 , label="Optimal Value")

        plt.xlabel("Planning Budget (log scale)", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        #plt.title(f"{metric} vs Planning Budget (Comparison)")
        #plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.savefig(f"comparison_{metric.replace(' ', '_').lower()}.png")
        plt.show()

if __name__ == "__main__":

    map_size = 8
    TRAIN_CONFIG = "NO_HOLES"
    CONFIG = "SLALOM"
    vfunc = "nn"

    # Example usage
    filepaths = (
        [   
            # f"final/{map_size}x{map_size}/az/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_ValueEst_({vfunc})_{map_size}x{map_size}_{CONFIG}.csv",
            # f"final/{map_size}x{map_size}/az/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_ValueEst_({vfunc})_{map_size}x{map_size}_{CONFIG}.csv",
            f"{map_size}x{map_size}/CCW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(1.0)_ValueEst_({vfunc})_{map_size}x{map_size}_{TRAIN_CONFIG}_{CONFIG}.csv",
            f"{map_size}x{map_size}/CCW_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(1.0)_ValueEst_({vfunc})_{map_size}x{map_size}_{TRAIN_CONFIG}_{CONFIG}.csv",

            # f"{map_size}x{map_size}/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(UCT)_c_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_({vfunc})_UpdateEst_(True)_{map_size}x{map_size}_{CONFIG}.csv",
            #f"{map_size}x{map_size}/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(PUCT)_c_(1.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_({vfunc})_UpdateEst_(True)_{map_size}x{map_size}_{CONFIG}.csv",

            # f"final/{map_size}x{map_size}/mvc/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_ValueEst_({vfunc})_{map_size}x{map_size}_{CONFIG}.csv",
            # f"final/{map_size}x{map_size}/mvc/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_ValueEst_({vfunc})_{map_size}x{map_size}_{CONFIG}.csv",
            f"{map_size}x{map_size}/CCW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(1.0)_ValueEst_({vfunc})_{map_size}x{map_size}_{TRAIN_CONFIG}_{CONFIG}.csv",
            f"{map_size}x{map_size}/CCW_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(1.0)_ValueEst_({vfunc})_{map_size}x{map_size}_{TRAIN_CONFIG}_{CONFIG}.csv",
            #f"final/{map_size}x{map_size}/Algorithm_(mini-trees)_EvalPol_(visit)_SelPol_(UCT)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(True)_ValueEst_({vfunc})_UpdateEst_(True)_{map_size}x{map_size}_{CONFIG}.csv"
            #f"{map_size}x{map_size}/Algorithm_(mega-tree)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Predictor_(current_value)_n_(4)_eps_(0.05)_ValueSearch_(False)_ValueEst_({vfunc})_UpdateEst_(True)_{map_size}x{map_size}_{CONFIG}.csv"
            # f"{map_size}x{map_size}/Algorithm_(octopus)_EvalPol_(qt_max)_SelPol_(PolicyUCT)_c_(0.0)_Predictor_(current_value)_eps_(0.05)_ValueEst_({vfunc})_(True)_1_{map_size}x{map_size}_{CONFIG}.csv",
            # f"{map_size}x{map_size}/Algorithm_(octopus)_EvalPol_(qt_max)_SelPol_(PolicyUCT)_c_(0.0)_Predictor_(current_value)_eps_(0.05)_ValueEst_({vfunc})_(True)_Value_Penalty_1_{map_size}x{map_size}_{CONFIG}.csv"
            # f"{map_size}x{map_size}/Algorithm_(octopus)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Predictor_(current_value)_eps_(0.05)_ValueEst_({vfunc})_(True)_ttemp_(0.0)_Value_Penalty_1_{map_size}x{map_size}_{TRAIN_CONFIG}_{CONFIG}.csv"

            # f"{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_ValueEst_({vfunc})_{map_size}x{map_size}_{TRAIN_CONFIG}_{CONFIG}.csv",
            # f"{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.5)_ValueEst_({vfunc})_{map_size}x{map_size}_{TRAIN_CONFIG}_{CONFIG}.csv",
            # f"final/{map_size}x{map_size}/mvc/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(1.0)_ValueEst_({vfunc})_{map_size}x{map_size}_{TRAIN_CONFIG}_{CONFIG}.csv",
            # f"{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(2.0)_ValueEst_({vfunc})_{map_size}x{map_size}_{TRAIN_CONFIG}_{CONFIG}.csv",
            # f"{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(100.0)_ValueEst_({vfunc})_{map_size}x{map_size}_{TRAIN_CONFIG}_{CONFIG}.csv",

            # f"{map_size}x{map_size}/Algorithm_(octopus)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Predictor_(current_value)_eps_(0.05)_ValueEst_({vfunc})_(True)_ttemp_(0.0)_Value_Penalty_1_{map_size}x{map_size}_{TRAIN_CONFIG}_{CONFIG}_NO_REUSE.csv",
            # f"{map_size}x{map_size}/Algorithm_(octopus)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(1.0)_Predictor_(current_value)_eps_(0.05)_ValueEst_({vfunc})_(True)_ttemp_(0.0)_Value_Penalty_1_{map_size}x{map_size}_{TRAIN_CONFIG}_{CONFIG}_C>0.csv",
            # f"{map_size}x{map_size}/Algorithm_(octopus)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Predictor_(current_value)_eps_(0.05)_ValueEst_({vfunc})_(True)_ttemp_(0.0)_Value_Penalty_0.0_{map_size}x{map_size}_{TRAIN_CONFIG}_{CONFIG}_NO_VP.csv",
            # f"{map_size}x{map_size}/Algorithm_(octopus)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Predictor_(current_value)_eps_(0.05)_ValueEst_({vfunc})_(True)_ttemp_(None)_Value_Penalty_1_{map_size}x{map_size}_{TRAIN_CONFIG}_{CONFIG}_NONE_TEMP.csv",

            f"{map_size}x{map_size}/CCW_Algorithm_(octopus)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.0)_Predictor_(current_value)_eps_(0.05)_ValueEst_({vfunc})_(True)_ttemp_(0.0)_Value_Penalty_1_{map_size}x{map_size}_{TRAIN_CONFIG}_{CONFIG}.csv"

        ]
        #16x16/Algorithm_(octopus)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0)_Predictor_(current_value)_eps_(0.05)_ValueEst_(perfect)_(True)_16x16_NARROW_XTREME.csv
    )
    labels = ["AZ+UCT", "AZ+PUCT" , "MVC+UCT", "MVC+PUCT" ,"CAP"]
    #labels = ["MVC+UCT c=0.0", "MVC+UCT c=0.5", "MVC+UCT c=1.0", "MVC+UCT c=2.0", "MVC+UCT c=100.0", "CAP"]
    #labels = ["NO TREE REUSE", "C>0", "NO VP", "NONE TEMP", "STANDARD"]


    plot_comparison_from_csvs(filepaths, labels, map_size, 100)