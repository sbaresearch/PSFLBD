import pandas as pd

# Load data
mnist = pd.read_csv("all_results_mnist.csv")
cifar = pd.read_csv("all_results_cifar10.csv")
gtsrb = pd.read_csv("all_results_gtsrb.csv")

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
    "attack_mode": "MR"
})

# Map MR values
df["MR"] = df["MR"].map({
    "baseline": "no",
    "replacement": "yes"
})

# Pivot to wide format
pivot_df = df.pivot_table(
    index=["Dataset", "FS", "MS", "MR"],
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
    "DBA_ACC": "DBA_Acc",
    "neurotoxin_ACC": "Neurotoxin_Acc",
    "neurotoxin_ASR": "Neurotoxin_ASR",
    "neurotoxin_RA": "Neurotoxin_RA",
})

# Reorder columns
final_cols = [
    "Dataset", "FS", "MS", "MR",
    "badnet_ASR", "badnet_Acc", "badnet_RA",
    "DBA_ASR", "DBA_Acc", "DBA_RA",
    "Neurotoxin_ASR", "Neurotoxin_Acc", "Neurotoxin_RA"
]

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

pivot_df = pivot_df.sort_values(by=["Dataset", "MR", "FS", "MS"]).reset_index(drop=True)

# Save
pivot_df.to_csv("all_results_by_ds.csv", index=False)

print("Saved to all_results_by_ds.csv")




# for LATEX


# Load final dataset
df = pd.read_csv("all_results_by_ds.csv")

# Replace NaN / missing with "-"
df = df.fillna("-")

# Define custom orderings
fs_order = ["avg", "mkrum", "foolsgold", "mmetric", "alignins", "feddlad"]
ms_order = ["avg", "mkrum", "foolsgold", "mmetric", "alignins", "feddlad"]

dataset_order = ["MNIST", "CIFAR-10", "GTSRB"]
mr_order = ["no", "yes"]

# Convert to categorical for sorting
df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)
df["FS"] = pd.Categorical(df["FS"], categories=fs_order, ordered=True)
df["MS"] = pd.Categorical(df["MS"], categories=ms_order, ordered=True)
df["MR"] = pd.Categorical(df["MR"], categories=mr_order, ordered=True)

# Sort
df = df.sort_values(by=["Dataset", "MR", "MS", "FS"]).reset_index(drop=True)

# Output file
out_file = "all_results_by_ds.txt"

with open(out_file, "w") as f:
    # Header
    header = " & ".join(df.columns) + " \\\\"
    f.write(header + "\n")

    # Rows
    for _, row in df.iterrows():
        line = " & ".join(str(x) for x in row.values) + " \\\\"
        f.write(line + "\n")

print(f"Saved LaTeX-formatted table to {out_file}")