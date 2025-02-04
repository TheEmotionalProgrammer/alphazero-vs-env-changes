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
            
            # Plot mean values
            plt.plot(df["Budget"], df[f"{metric} mean"], marker="o", linestyle="-", label=label)

            # Fill shaded area for standard error
            plt.fill_between(df["Budget"], 
                             df[f"{metric} mean"] - df[f"{metric} SE"], 
                             df[f"{metric} mean"] + df[f"{metric} SE"], 
                             alpha=0.1)

        # Set x-axis to logarithmic scale
        plt.xscale("log", base=2)

        # Set x-ticks explicitly
        plt.xticks(dataframes[0]["Budget"], labels=dataframes[0]["Budget"])

        # Adjust y-axis limits dynamically
        plt.ylim(adjust_ylim(metric))

        # Add a horizontal reference line for optimal values (if applicable)
        if metric in optimal_values:
            plt.axhline(optimal_values[metric], color='red', linestyle='dotted', linewidth=1.5, label="Optimal Value")

        plt.xlabel("Planning Budget (log scale)")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Planning Budget (Comparison)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.savefig(f"comparison_{metric.replace(' ', '_').lower()}.png")
        plt.show()

if __name__ == "__main__":
    # Example usage
    filepaths = (
        ["8x8/Algorithm_(azdetection)_EvalPol_(mvc)_SelPol_(PolicyUCT)_Predictor_(current_value)_n_(4)_eps_(0.01)_PlanningStyle_(connected)_ValueSearch_(False)_DEAD_END.csv",
         "8x8/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_DEAD_END.csv"
        ]
    )
    labels = ["AZDetection", "AZMCTS"]
    plot_comparison_from_csvs(filepaths, labels, 8, 30)