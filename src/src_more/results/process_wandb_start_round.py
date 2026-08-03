import wandb
import pandas as pd

import argparse
import csv
from pathlib import Path

api = wandb.Api()

# Define your project path and the group name you want to fetch
# Path format: "entity/project-name"
project_paths = [
    "adavarts-tuwien/sfl-with-start-round",
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
        if run.config['defense_start_round'] == 1:
            # print(f"Skipping run {run.id} because defense_start_round is 1")
            continue
        dsr = run.config['defense_start_round']
        print(f"Run {run.id} with defense_start_round {dsr}")
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
combined_df.to_csv(f"all_start_rounds.csv", index=False)

print(f"Successfully exported {len(all_runs_data)} runs to all_start_rounds.csv")




def pick_first_existing(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    print(f"Warning: None of the candidates {candidates} were found in columns {columns}")
    return None


def is_non_empty(value):
    return value is not None and str(value).strip() != ""


def last_non_empty(rows, column_name):
    for row in reversed(rows):
        value = row.get(column_name, "")
        if is_non_empty(value):
            return value
    return ""


def sort_key(row):
    def as_float(name):
        value = row.get(name, "")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf")

    return (
        row.get("run_id", ""),
        as_float("round"),
        as_float("_step"),
        as_float("_timestamp"),
    )


def build_final_rows(rows):
    if not rows:
        return []

    columns = set(rows[0].keys())
    metric_cols = {
        "asr": pick_first_existing(columns, ["asr", "ASR"]),
        "main_task": pick_first_existing(columns, ["clean_acc", "main_task", "main task", "accuracy"]),
        "robustness": pick_first_existing(columns, ["robustness acc", "robustness", "robust_acc"]),
        "rounds": pick_first_existing(columns, ["rounds", "round"]),
        "defense_start_round": pick_first_existing(columns, ["config_defense_start_round", "defense_start_round"]),
    }

    if metric_cols["asr"] is None:
        raise ValueError("Could not find an ASR column in the input CSV.")
    if metric_cols["main_task"] is None:
        raise ValueError("Could not find a clean_acc/main_task column in the input CSV.")
    if metric_cols["robustness"] is None:
        raise ValueError("Could not find a robustness column in the input CSV.")

    group_key = "run_id" if "run_id" in columns else "run_name"
    if group_key not in columns:
        raise ValueError("Input CSV must contain either 'run_id' or 'run_name'.")

    rows = sorted(rows, key=sort_key)
    grouped = {}
    for row in rows:
        grouped.setdefault(row[group_key], []).append(row)

    final_rows = []
    for _, run_rows in grouped.items():
        last_row = run_rows[-1]
        final_rows.append(
            {
                "run_id": last_row.get("run_id", ""),
                "run_name": last_row.get("run_name", ""),
                "dataset": last_row.get("config_data", ""),
                "attack": last_row.get("config_attack", ""),
                "attack_mode": last_row.get("config_attack_mode", ""),
                "defense": last_row.get("config_aggr", ""),
                "defense_server": last_row.get("config_aggr_server", ""),
                "defense_start_round": last_non_empty(run_rows, metric_cols["defense_start_round"]),
                "rounds": last_non_empty(run_rows, metric_cols["rounds"]),
                "round": last_non_empty(run_rows, metric_cols["rounds"]),
                "bg": last_row.get("config_bg", ""),
                "pg": last_row.get("config_pg", ""),
                "n_neighbors": last_row.get("config_n_neighbors", ""),
                "iqr_scale": last_row.get("config_iqr_scale", ""),
                "asr": last_non_empty(run_rows, metric_cols["asr"]),
                "main_task": last_non_empty(run_rows, metric_cols["main_task"]),
                "robustness": last_non_empty(run_rows, metric_cols["robustness"]),
            }
        )

    return final_rows


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def pick_best_rows(rows):
    grouped = {}
    for row in rows:
        key = (
            row.get("dataset", ""),
            row.get("attack", ""),
            row.get("attack_mode", ""),
            row.get("defense", ""),
            row.get("defense_server", ""),
            row.get("defense_start_round", ""),
            row.get("rounds", ""),
        )
        grouped.setdefault(key, []).append(row)

    best_rows = []
    for _, group_rows in grouped.items():
        best_row = max(
            group_rows,
            key=lambda row: (
                as_float(row.get("asr", "")),
                as_float(row.get("main_task", "")),
                as_float(row.get("robustness", "")),
            ),
        )
        best_rows.append(best_row)

    return best_rows


def filter_rows(rows):
    filtered = []
    for row in rows:
        attack = row.get("attack", "")
        attack_mode = row.get("attack_mode", "")

        if attack == "pgd" or attack_mode == "pgd":
            continue

        filtered.append(row)

    return filtered


input_path = Path("all_start_rounds.csv")
output_path = Path("processed_start_rounds.csv")

with input_path.open(newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Read {len(rows)} rows from {input_path}")
final_rows = build_final_rows(rows)
print(f"Built {len(final_rows)} final rows after collapsing history")
final_rows = pick_best_rows(final_rows)
print(f"Picked {len(final_rows)} best rows per (dataset, attack, defense) combination")
final_rows = filter_rows(final_rows)
print(f"{len(final_rows)} rows remain after filtering out PGD and non-BadNet replacement attacks")
fieldnames = [
    "run_id",
    "run_name",
    "dataset",
    "attack",
    "attack_mode",
    "defense",
    "defense_server",
    "defense_start_round",
    "rounds",
    "round",
    "bg",
    "pg",
    "n_neighbors",
    "iqr_scale",
    "asr",
    "main_task",
    "robustness",
]

with output_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(final_rows)

print(f"Wrote {len(final_rows)} final rows to {output_path}")




OUTPUT_COLUMNS = [
    "attack",
    "attack_mode",
    "defense",
    "defense_server",
    "defense_start_round",
    "rounds",
    "asr",
    "main_task",
    "robustness",
]


def as_float(value, default=float("inf")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_percent(value):
    number = as_float(value, default=None)
    if number is None:
        return ""
    return f"{number * 100:.2f}"


def best_row(rows):
    # Lower ASR is better; for ties prefer higher clean and robust accuracy.
    return min(
        rows,
        key=lambda row: (
            as_float(row.get("asr", "")),
            -as_float(row.get("main_task", ""), default=float("-inf")),
            -as_float(row.get("robustness", ""), default=float("-inf")),
        ),
    )


def reduce_rows(rows):
    grouped = {}
    for row in rows:
        key = (
            row.get("attack", ""),
            row.get("attack_mode", ""),
            row.get("defense", ""),
            row.get("defense_server", ""),
            row.get("defense_start_round", ""),
            row.get("rounds", ""),
        )
        grouped.setdefault(key, []).append(row)

    reduced = []
    for key, group in grouped.items():
        _, _, _, defense_server, _, _ = key
        chosen = best_row(group) if defense_server == "feddlad" else best_row(group)
        output_row = {column: chosen.get(column, "") for column in OUTPUT_COLUMNS}
        for metric in ("asr", "main_task", "robustness"):
            output_row[metric] = format_percent(chosen.get(metric, ""))
        reduced.append(output_row)

    return sorted(
        reduced,
        key=lambda row: (
            row["attack"],
            row["attack_mode"],
            row["defense"],
            row["defense_server"],
            row["defense_start_round"],
            row["rounds"],
        ),
    )


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

mnist_output_path = Path("all_start_rounds_mnist.csv")
cifar10_output_path = Path("all_start_rounds_cifar10.csv")
gtsrb_output_path = Path("all_start_rounds_gtsrb.csv")
rows = final_rows

mnist_rows = reduce_rows([row for row in rows if row.get("dataset", "") == "mnist"])
cifar10_rows = reduce_rows([row for row in rows if row.get("dataset", "") == "cifar10"])
gtsrb_rows = reduce_rows([row for row in rows if row.get("dataset", "") == "gtsrb"])

write_csv(mnist_output_path, mnist_rows)
write_csv(cifar10_output_path, cifar10_rows)
write_csv(gtsrb_output_path, gtsrb_rows)

print(f"Wrote {len(mnist_rows)} rows to {mnist_output_path}")
print(f"Wrote {len(cifar10_rows)} rows to {cifar10_output_path}")
print(f"Wrote {len(gtsrb_rows)} rows to {gtsrb_output_path}")


mnist = pd.read_csv(mnist_output_path)
cifar = pd.read_csv(cifar10_output_path)
gtsrb = pd.read_csv(gtsrb_output_path)

# Add dataset column
mnist["Dataset"] = "MNIST"
cifar["Dataset"] = "CIFAR-10"
gtsrb["Dataset"] = "GTSRB"

# Combine
df = pd.concat([mnist, cifar, gtsrb], ignore_index=True)

# Rename for clarity
df = df.rename(columns={
    "defense": "FS",
    "defense_server": "MS",
    "main_task": "Acc",
    "robustness": "RA",
    "defense_start_round": "defense_start_round",
    "rounds": "Rounds",
})

# Pivot to wide format
pivot_df = df.pivot_table(
    index=["Dataset", "FS", "MS", "defense_start_round", "Rounds"],
    columns="attack",
    values=["asr", "Acc", "RA"]
)

# Flatten column names
pivot_df.columns = [
    f"{attack}_{metric.upper()}" 
    for metric, attack in pivot_df.columns
]

pivot_df = pivot_df.reset_index()

# Rename columns to match required format
pivot_df = pivot_df.rename(columns={
    "badnet_ACC": "badnet_Acc",
    "neurotoxin_ACC": "Neurotoxin_Acc",
    "neurotoxin_ASR": "Neurotoxin_ASR",
    "neurotoxin_RA": "Neurotoxin_RA",
})

# Reorder columns
final_cols = [
    "Dataset", "FS", "MS", "defense_start_round", "Rounds",
    "badnet_ASR", "badnet_Acc", "badnet_RA",
    "Neurotoxin_ASR", "Neurotoxin_Acc", "Neurotoxin_RA"
]
print("Final columns:", final_cols)
# 👉 Add missing columns as "-"
for col in final_cols:
    if col not in pivot_df.columns:
        pivot_df[col] = "-"

pivot_df = pivot_df[final_cols]

# Enforce dataset order
pivot_df["Dataset"] = pd.Categorical(
    pivot_df["Dataset"],
    categories=["MNIST", "CIFAR-10", "GTSRB"],
    ordered=True
)

pivot_df = pivot_df.sort_values(by=["Dataset", "defense_start_round", "Rounds", "FS", "MS"]).reset_index(drop=True)

# Save
pivot_df.to_csv("all_results_rounds.csv", index=False)

print("Saved to all_results_rounds.csv")




# for LATEX


# Load final dataset
df = pd.read_csv("all_results_rounds.csv")

# Replace NaN / missing with "-"
df = df.fillna("-")

# Define custom orderings
fs_order = ["avg", "mkrum", "foolsgold", "alignins", "feddlad"]
ms_order = ["avg", "mkrum", "foolsgold", "alignins", "feddlad"]

dataset_order = ["MNIST", "CIFAR-10", "GTSRB"]

# Convert to categorical for sorting
df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)
df["FS"] = pd.Categorical(df["FS"], categories=fs_order, ordered=True)
df["MS"] = pd.Categorical(df["MS"], categories=ms_order, ordered=True)

# Sort
df = df.sort_values(by=["Dataset", "defense_start_round", "Rounds", "FS", "MS"]).reset_index(drop=True)

# Output file
out_file = "all_results_rounds.txt"

with open(out_file, "w") as f:
    # Header
    header = " & ".join(df.columns) + " \\\\"
    f.write(header + "\n")

    # Rows
    for _, row in df.iterrows():
        line = " & ".join(str(x) for x in row.values) + " \\\\"
        f.write(line + "\n")

print(f"Saved LaTeX-formatted table to {out_file}")