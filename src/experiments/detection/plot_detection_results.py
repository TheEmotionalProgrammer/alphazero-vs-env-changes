import json
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":   
    
    nn_current_value_true = json.load(open("detection_results/value_estimate=nn__predictor=current_value__update_estimator=True.json", "r"))
    nn_current_value_false = json.load(open("detection_results/value_estimate=nn__predictor=current_value__update_estimator=False.json", "r"))
    nn_original_env = json.load(open("detection_results/value_estimate=nn__predictor=original_env__update_estimator=False.json", "r"))

    train_seeds = 10 

    # Compute the errors
    errors_nn_current_value_false = {}
    errors_nn_current_value_true = {}

    for env in nn_original_env:
        errors_nn_current_value_false[env] = {}
        errors_nn_current_value_true[env] = {}

        for seed in range(train_seeds):
            seed = str(seed)
            
            if nn_original_env[env][seed]["distance_init_obst"] is None:
                errors_nn_current_value_false[env][seed] = 0 if nn_current_value_false[env][seed]["distance_init_obst"] is None else float("inf")
                errors_nn_current_value_true[env][seed] = 0 if nn_current_value_true[env][seed]["distance_init_obst"] is None else float("inf")
                continue

            errors_nn_current_value_false[env][seed] = nn_current_value_false[env][seed]["distance_init_obst"] - nn_original_env[env][seed]["distance_init_obst"]
            errors_nn_current_value_true[env][seed] = nn_current_value_true[env][seed]["distance_init_obst"] - nn_original_env[env][seed]["distance_init_obst"]
    
    # Compute the mean and standard deviation of errors
    mean_errors_nn_current_value_false = {env: np.mean(list(errors_nn_current_value_false[env].values())) for env in errors_nn_current_value_false}
    mean_errors_nn_current_value_true = {env: np.mean(list(errors_nn_current_value_true[env].values())) for env in errors_nn_current_value_true}
    std_errors_nn_current_value_false = {env: np.std(list(errors_nn_current_value_false[env].values())) for env in errors_nn_current_value_false}
    std_errors_nn_current_value_true = {env: np.std(list(errors_nn_current_value_true[env].values())) for env in errors_nn_current_value_true}
    
    custom_labels = ["Empty", "D3", "D6", "D9", "D12", "D15"]

    # Plot Accuracy Error with standard deviation as shaded area
    plt.figure()
    envs = list(mean_errors_nn_current_value_false.keys())
    mean_false = list(mean_errors_nn_current_value_false.values())
    std_false = list(std_errors_nn_current_value_false.values())
    mean_true = list(mean_errors_nn_current_value_true.values())
    std_true = list(std_errors_nn_current_value_true.values())

    plt.plot(envs, mean_false, label="standard")
    plt.fill_between(envs, np.array(mean_false) - np.array(std_false), np.array(mean_false) + np.array(std_false), alpha=0.2)
    plt.plot(envs, mean_true, label="$y^{max}$")
    plt.fill_between(envs, np.array(mean_true) - np.array(std_true), np.array(mean_true) + np.array(std_true), alpha=0.2)
    plt.title("Accuracy Error")
    plt.legend()
    plt.xticks(np.arange(len(custom_labels)), custom_labels)
    plt.xlabel("Configuration")
    plt.grid(True)
    plt.savefig("accuracy_error.png")

    # Compute Sensitivity Errors
    errors_nn_current_value_false = {}
    errors_nn_current_value_true = {}

    for env in nn_original_env:
        errors_nn_current_value_false[env] = {}
        errors_nn_current_value_true[env] = {}

        for seed in range(train_seeds):
            seed = str(seed)
            
            if nn_original_env[env][seed]["distance_from_init"] == "inf":
                errors_nn_current_value_false[env][seed] = 0 if nn_current_value_false[env][seed]["distance_from_init"] == "inf" else float("inf")
                errors_nn_current_value_true[env][seed] = 0 if nn_current_value_true[env][seed]["distance_from_init"] == "inf" else float("inf")
                continue

            errors_nn_current_value_false[env][seed] = nn_current_value_false[env][seed]["distance_from_init"] - nn_original_env[env][seed]["distance_from_init"]
            errors_nn_current_value_true[env][seed] = nn_current_value_true[env][seed]["distance_from_init"] - nn_original_env[env][seed]["distance_from_init"]
    
    # Compute the mean and standard deviation of errors
    mean_errors_nn_current_value_false = {env: np.mean(list(errors_nn_current_value_false[env].values())) for env in errors_nn_current_value_false}
    mean_errors_nn_current_value_true = {env: np.mean(list(errors_nn_current_value_true[env].values())) for env in errors_nn_current_value_true}
    std_errors_nn_current_value_false = {env: np.std(list(errors_nn_current_value_false[env].values())) for env in errors_nn_current_value_false}
    std_errors_nn_current_value_true = {env: np.std(list(errors_nn_current_value_true[env].values())) for env in errors_nn_current_value_true}

    # Plot Sensitivity Error with standard deviation as shaded area
    plt.figure()
    envs = list(mean_errors_nn_current_value_false.keys())
    mean_false = list(mean_errors_nn_current_value_false.values())
    std_false = list(std_errors_nn_current_value_false.values())
    mean_true = list(mean_errors_nn_current_value_true.values())
    std_true = list(std_errors_nn_current_value_true.values())

    plt.plot(envs, mean_false, label="standard")
    plt.fill_between(envs, np.array(mean_false) - np.array(std_false), np.array(mean_false) + np.array(std_false), alpha=0.2)
    plt.plot(envs, mean_true, label="$y^{max}$")
    plt.fill_between(envs, np.array(mean_true) - np.array(std_true), np.array(mean_true) + np.array(std_true), alpha=0.2)
    plt.title("Sensitivity Error")
    plt.legend()
    plt.xticks(np.arange(len(custom_labels)), custom_labels)
    plt.xlabel("Configuration")
    plt.grid(True)
    plt.savefig("sensitivity_error.png")
