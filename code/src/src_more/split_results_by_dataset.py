import argparse
import csv
from pathlib import Path
import pandas as pd

OUTPUT_COLUMNS = [
    "attack",
    "attack_mode",
    "defense",
    "defense_server",
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
        )
        grouped.setdefault(key, []).append(row)

    reduced = []
    for key, group in grouped.items():
        _, _, _, defense_server = key
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
        ),
    )


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Split wanted_results_final.csv into dataset-specific CSV files."
    )
    parser.add_argument(
        "--input",
        default="wanted_results_final.csv",
        help="Path to the processed final-results CSV.",
    )
    parser.add_argument(
        "--mnist-output",
        default="wanted_results_mnist.csv",
        help="Output path for MNIST rows.",
    )
    parser.add_argument(
        "--cifar10-output",
        default="wanted_results_cifar10.csv",
        help="Output path for CIFAR10 rows.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    mnist_rows = reduce_rows([row for row in rows if row.get("dataset", "") == "mnist"])
    cifar10_rows = reduce_rows([row for row in rows if row.get("dataset", "") == "cifar10"])

    write_csv(Path(args.mnist_output), mnist_rows)
    write_csv(Path(args.cifar10_output), cifar10_rows)

    print(f"Wrote {len(mnist_rows)} rows to {args.mnist_output}")
    print(f"Wrote {len(cifar10_rows)} rows to {args.cifar10_output}")

def preprocess_csv(csv_path):
    df = pd.read_csv(csv_path)
    df_colums = df.columns
    df_columns = [col.strip() for col in df_colums]
    df_columns[0] = "Attack"
    # df_columns[1] = "Model Replacement"
    # df_columns[3] = "ASR"
    # df_columns[4] = "Main Task Accuracy"
    # df_columns[5] = "Robustness Accuracy"
    # df_columns[2] = "Defense"
    df.columns = df_columns
    # df["Model Replacement"] = df["Model Replacement"].apply(lambda x: "Yes" if x == "replacement" else "No")
    # df["Defense"] = df["Defense"].replace(
    # {
    #     "avg": "No Defense",
    #     "feddlad": "FedDLAD",
    #     "foolsgold": "FoolsGold",
    #     "mmetric": "MMMetric",
    #     "mkrum": "MKrum",        
    #     "alignins": "AlignIns",
    #     # "neurotoxin": "Neurotoxin",
    # }
    # )
    # df = df[df["Defense"]!="rlr"].reset_index(drop=True)

    # df.drop(columns=["defense"], inplace=True)
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    # main()
    preprocess_csv("wanted_results_cifar10.csv")
