import argparse
import csv
from pathlib import Path


def pick_first_existing(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
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

        # if attack_mode == "replacement" and attack not in {"badnet", "DBA"}:
        #     continue

        filtered.append(row)

    return filtered


def main():
    parser = argparse.ArgumentParser(description="Collapse WandB history CSV to one final-results row per run.")
    parser.add_argument(
        "--input",
        default="all_results.csv",
        help="Path to the raw exported CSV from process_wandb.py",
    )
    parser.add_argument(
        "--output",
        default="all_results_final.csv",
        help="Path to the processed final-results CSV",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

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


if __name__ == "__main__":
    main()
