import wandb
import pandas as pd

api = wandb.Api()

# Define your project path and the group name you want to fetch
# Path format: "entity/project-name"
project_paths = [
    "adavarts-tuwien/fl-backdoors-code_src_src_more",
    "adavarts-tuwien/split-fed-backdoor-3",
    "adavarts-tuwien/sfl2"
]
# group_name = "CIFAR10"

all_runs_data = []

for project_path in project_paths:
    # Query the runs in that group
    runs = api.runs(project_path)  # , filters={"group": group_name}
    print(f"Fetched {len(runs)} runs from {project_path}")

    for run in runs:
        # .history() gets the logged metrics
        # .summary contains final values (like best accuracy)
        # .config contains your hyperparameters
        df = run.history()
        
        # Add metadata so you know which run is which
        df['run_id'] = run.id
        df['run_name'] = run.name
        
        # Optional: Add config values (like learning_rate) to every row
        for key, value in run.config.items():
            df[f"config_{key}"] = value
            
        all_runs_data.append(df)
    
    print(f"Processed {len(all_runs_data)}")

# Combine everything into one master table
combined_df = pd.concat(all_runs_data, ignore_index=True)
combined_df.to_csv(f"all_results.csv", index=False)

print(f"Successfully exported {len(all_runs_data)} runs to all_results.csv")