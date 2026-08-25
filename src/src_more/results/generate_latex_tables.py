"""
Render the paper's LaTeX result rows straight from `seed_results_aggregated.csv`
(produced by `extract_seed_results.py`). Cells with 2+ seeds are formatted as
mean^std (superscript notation, 1 decimal); cells still on a single seed are
marked with a trailing `*` so they're visibly not-yet-final.

Two output shapes:

1. table{1,2,3}_*_rows.tex -- body rows only (no caption/label/tabular
   wrapper) matching the paper's actual multirow table layout, ready to
   paste between \\toprule and \\bottomrule of the existing tables.
2. seed_results_by_ds.{txt,csv} -- flat pivot (Dataset & FS & MS & MR &
   badnet_ASR & ... \\), covering every (dataset, FS, MS, MR) combo present
   in the data.
"""
import csv
import re
from pathlib import Path

AGG_PATH = Path(__file__).parent / "seed_results_aggregated.csv"
OUT_DIR = Path(__file__).parent

ATTACKS = ["badnet", "DBA", "neurotoxin"]
ATTACK_COL_LABEL = {"badnet": "badnet", "DBA": "DBA", "neurotoxin": "Neurotoxin"}
MODES = [("baseline", "no"), ("replacement", "yes")]
MODE_TO_MR = {"baseline": "no", "replacement": "yes"}
MODEL_FOR = {"gtsrb": "googlenet"}  # else resnet18
DATASET_LABEL = {"mnist": "MNIST", "cifar10": "CIFAR-10", "gtsrb": "GTSRB"}
DATASET_ORDER = ["MNIST", "CIFAR-10", "GTSRB"]
FS_MS_ORDER = ["avg", "mkrum", "foolsgold", "mmetric", "alignins", "feddlad"]
DEFENSE_LABEL = {
    "mkrum": "MKrum",
    "foolsgold": "FoolsGold",
    "mmetric": "MMetric",
    "alignins": "AlignIns",
    "feddlad": "FedDLAD",
}

# Metrics shown per attack in the table{1,2,3,4}_*.tex row generators, in
# column order.
TABLE_METRICS = ("asr", "acc", "ra")

# the "standard" experiment config -- excludes off-default variants (e.g. the
# num_agents=10 dev run seen during verification) that would otherwise create
# a second, conflicting row for the same nominal (dataset, FS, MS, MR).
STANDARD = {
    "bg": "3", "pg": "0", "n_neighbors": "0", "iqr_scale": "0.5",
    "num_agents": "20", "num_corrupt": "4", "poison_frac": "0.5",
    "defense_start_round": "1",
}


# Toggle to compare presentation styles -- not settled on one yet.
# "pm":   93.5$\pm$4.3
# "tiny": 93.5\tiny{$\pm$4.3}
# $ wraps only the \pm macro itself -- neither the mean nor the std digits
# are ever inside math mode.
STD_STYLE = "tiny"


def fmt1(pct_str):
    return f"{float(pct_str):.1f}" if pct_str else None


def format_mean_std(mean, std):
    if STD_STYLE == "tiny":
        return f"{mean}\\tiny{{$\\pm${std}}}"
    return f"{mean}$\\pm${std}"


def cell_text(row, metric):
    """metric in {'asr','acc','ra'}. Self-contained LaTeX fragment, or '-' if
    empty. Only n_seeds==3 (all of seed 0/43/44 present) gets the clean
    mean+-std -- that's the only "finished" state. n_seeds 1 or 2 still show
    the best current estimate (bare value for 1 seed, mean+-std for 2) but
    suffixed with one '*' per seed already present, so a glance at the star
    count tells you how many seeds are still missing before it's final."""
    n_seeds = int(row["n_seeds"])
    mean = fmt1(row[f"{metric}_mean_pct"])
    if mean is None:
        return "-"
    if n_seeds == 1:
        for s in (0, 43, 44):
            val = fmt1(row[f"seed{s}_{metric}_pct"])
            if val is not None:
                return f"{val}*"
        return "-"
    std = fmt1(row[f"{metric}_std_pct"])
    cell = format_mean_std(mean, std)
    if n_seeds < 3:
        cell += "*" * n_seeds
    return cell


def is_standard(row):
    model_ok = row["model"] == MODEL_FOR.get(row["data"], "resnet18")
    return model_ok and all(row.get(k) == v for k, v in STANDARD.items())


def load_rows():
    with AGG_PATH.open() as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Shape 1: paper-layout body rows (table1/2/3), looked up by full config key
# ---------------------------------------------------------------------------

def index_rows(rows):
    """key = (data, model, attack-or-CLEAN, attack_mode-or-None, aggr, aggr_server).

    Filtered to `is_standard()` rows only. Without this, configs that were
    also swept over a non-default field (e.g. defense_start_round in {1, 50,
    100} for some GTSRB both-defense combos) collapse onto the same 6-field
    key and whichever row happens to appear last in the CSV silently wins --
    that's how a defense_start_round=50 run once got displayed as if it were
    the standard defense_start_round=1 experiment."""
    idx = {}
    for r in rows:
        if r["attack"] != "CLEAN" and not is_standard(r):
            continue
        key = (r["data"], r["model"], r["attack"], r["attack_mode"] or None, r["aggr"], r["aggr_server"])
        idx[key] = r
    return idx


def cell_from_idx(idx, data, attack, attack_mode, aggr, aggr_server, metric):
    """metric in {'asr','acc','ra'}. Returns a ready-to-paste LaTeX fragment,
    or '-' if the config doesn't exist in the aggregated CSV at all."""
    model = MODEL_FOR.get(data, "resnet18")
    row = idx.get((data, model, attack, attack_mode, aggr, aggr_server))
    if row is None:
        return "-"
    return cell_text(row, metric)


def numeric_and_std_from_idx(idx, data, attack, attack_mode, aggr, aggr_server, metric):
    """(value, std) as rounded (1-decimal) floats, matching what cell_from_idx
    would display -- or (None, None) if that config has no data. std is None
    for single-seed cells (nothing to compare)."""
    model = MODEL_FOR.get(data, "resnet18")
    row = idx.get((data, model, attack, attack_mode, aggr, aggr_server))
    if row is None:
        return None, None
    val = fmt1(row[f"{metric}_mean_pct"])
    std = fmt1(row[f"{metric}_std_pct"])
    return (float(val) if val is not None else None), (float(std) if std is not None else None)


def pick_extreme(candidates, kind):
    """candidates: [(key, value, std), ...]. kind='min' or 'max'. Returns the
    winning key. Ties on value are broken by smaller std -- e.g. two defenses
    both at 0.3% ASR, one +-0.1 and one +-0.2, prefer the +-0.1 one since it's
    the more confident estimate. A candidate with no std (single-seed, so no
    variance estimate at all) loses every tie-break."""
    target = min(c[1] for c in candidates) if kind == "min" else max(c[1] for c in candidates)
    tied = [c for c in candidates if c[1] == target]
    if len(tied) > 1:
        tied.sort(key=lambda c: (c[2] is None, c[2]))
    return tied[0][0]


def apply_highlight(cell_str, kind):
    """'bold': wraps just the leading number (the mean) in \\textbf{} --
    \\textbf can't reach into the \\tiny{$\\pm$std} suffix anyway (math mode
    has its own font weight), so bolding only the mean is both simpler and
    the only part that would actually render bold.
    'red': wraps the WHOLE cell (mean + \\tiny{$\\pm$std} suffix) in
    \\fcolorbox{red}{white}{} -- \\fcolorbox just frames its argument, no
    math-mode restriction, so the box can cleanly enclose the full value.
    \\fboxsep is shrunk locally (default 3pt made the row visibly taller
    than its unhighlighted neighbors; 0pt made the frame hug too tightly) --
    1pt is a middle ground: a small but present margin.
    Leaves non-numeric cells ('-') untouched."""
    if kind is None:
        return cell_str
    if kind == "red":
        return f"{{\\setlength{{\\fboxsep}}{{1pt}}\\fcolorbox{{red}}{{white}}{{{cell_str}}}}}"
    m = re.match(r"-?\d+\.?\d*", cell_str)
    if not m:
        return cell_str
    num, rest = m.group(0), cell_str[m.end():]
    if kind == "bold":
        return f"\\textbf{{{num}}}{rest}"
    return cell_str


def best_worst_metric(idx, data, defenses, role, metric, lower_is_better):
    """{(attack, mode): (best_defense, worst_defense)} across `defenses`, for
    highlighting. For ASR (lower_is_better=True) best=lowest/worst=highest,
    matching the paper's "best/worst based on ASR" convention; for Acc
    (lower_is_better=False) it's flipped -- best=highest/worst=lowest. Ties
    broken by smaller std (see pick_extreme). Omits groups where every
    defense has the same value (nothing meaningful to highlight) or fewer
    than 2 have data."""
    best_kind, worst_kind = ("min", "max") if lower_is_better else ("max", "min")
    result = {}
    for attack in ATTACKS:
        for mode, mr in MODES:
            candidates = []
            for defense in defenses:
                aggr, aggr_server = (defense, "avg") if role == "CAS" else ("avg", defense)
                v, s = numeric_and_std_from_idx(idx, data, attack, mode, aggr, aggr_server, metric)
                if v is not None:
                    candidates.append((defense, v, s))
            if len(candidates) < 2:
                continue
            vals = [c[1] for c in candidates]
            if min(vals) == max(vals):
                continue
            result[(attack, mode)] = (pick_extreme(candidates, best_kind), pick_extreme(candidates, worst_kind))
    return result


def best_worst_metric_both(idx, data, defenses, mode, metric, lower_is_better):
    """{attack: (best_pair, worst_pair)} across the full CAS x MS grid (all
    defenses x defenses combos, pair=(cas_defense, ms_defense)), for Table 4
    highlighting -- there's no MR split there (single `mode`), so this
    compares across all 16 rows for a given attack rather than per-defense
    like `best_worst_metric` does. Ties broken by smaller std (pick_extreme)."""
    best_kind, worst_kind = ("min", "max") if lower_is_better else ("max", "min")
    result = {}
    for attack in ATTACKS:
        candidates = []
        for cas_defense in defenses:
            for ms_defense in defenses:
                v, s = numeric_and_std_from_idx(idx, data, attack, mode, cas_defense, ms_defense, metric)
                if v is not None:
                    candidates.append(((cas_defense, ms_defense), v, s))
        if len(candidates) < 2:
            continue
        vals = [c[1] for c in candidates]
        if min(vals) == max(vals):
            continue
        result[attack] = (pick_extreme(candidates, best_kind), pick_extreme(candidates, worst_kind))
    return result


LOWER_IS_BETTER = {"asr": True, "acc": False, "ra": False}
# Acc/RA only ever get the bold "best" treatment -- a low-accuracy defense
# isn't "worst" the way a high-ASR one clearly is, so no red box for them.
RED_METRICS = {"asr"}


def render_table1_rows(idx, datasets=("mnist", "cifar10", "gtsrb")):
    """Table 1 body rows only (baseline attacks, aggr=avg, aggr_server=avg).
    No caption/label/tabular wrapper -- paste directly between \\toprule and
    \\bottomrule of the existing table."""
    lines = []
    for di, data in enumerate(datasets):
        label = DATASET_LABEL.get(data, data)
        for ri, (mode, mr) in enumerate(MODES):
            row_cells = []
            for attack in ATTACKS:
                for metric in TABLE_METRICS:
                    row_cells.append(cell_from_idx(idx, data, attack, mode, "avg", "avg", metric))
            prefix = f"\\multirow{{2}}{{*}}{{{label}}}" if ri == 0 else ""
            lines.append(f"{prefix}& {mr} & " + " & ".join(row_cells) + r" \\")
        if di < len(datasets) - 1:
            lines.append(r"\midrule")
    return "\n".join(lines)


def render_defense_table_rows(idx, defenses, role, datasets=("mnist", "cifar10", "gtsrb"), highlight_datasets=()):
    """Table 2/3 body rows only (role='CAS': aggr varies, aggr_server=avg fixed;
    role='MS': aggr=avg fixed, aggr_server varies). No caption/label/tabular
    wrapper -- paste directly between \\toprule and \\bottomrule.

    For datasets listed in `highlight_datasets`, each metric column in
    TABLE_METRICS is bolded (best) / red-boxed (worst) per (attack, MR) group
    across `defenses` -- MR=no and MR=yes are never compared against each
    other. ASR: best=lowest/worst=highest; Acc: best=highest/worst=lowest
    (see LOWER_IS_BETTER). Datasets not listed are left plain (e.g.
    CIFAR-10's CAS numbers are too close together for highlighting to mean
    anything)."""
    assert role in ("CAS", "MS")
    lines = []
    # last column index = data(1) + cas(1) + ms(1) + mr(1) + attacks*metrics
    # (span starts at 2, but the index itself still counts the data column)
    last_col = 4 + len(ATTACKS) * len(TABLE_METRICS)
    for di, data in enumerate(datasets):
        dlabel = DATASET_LABEL.get(data, data)
        extrema = {
            metric: best_worst_metric(idx, data, defenses, role, metric, LOWER_IS_BETTER[metric])
            for metric in TABLE_METRICS
        } if data in highlight_datasets else {}
        lines.append(rf"\multirow{{{2*len(defenses)}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{dlabel}}}}}")
        for defi, defense in enumerate(defenses):
            aggr, aggr_server = (defense, "avg") if role == "CAS" else ("avg", defense)
            label = DEFENSE_LABEL.get(defense, defense)
            cas_col, ms_col = (label, "-") if role == "CAS" else ("-", label)
            for ri, (mode, mr) in enumerate(MODES):
                row_cells = []
                for attack in ATTACKS:
                    for metric in TABLE_METRICS:
                        cell = cell_from_idx(idx, data, attack, mode, aggr, aggr_server, metric)
                        if (attack, mode) in extrema.get(metric, {}):
                            best_def, worst_def = extrema[metric][(attack, mode)]
                            if defense == best_def:
                                cell = apply_highlight(cell, "bold")
                            elif defense == worst_def and metric in RED_METRICS:
                                cell = apply_highlight(cell, "red")
                        row_cells.append(cell)
                if ri == 0:
                    prefix = f"& \\multirow{{2}}{{*}}{{{cas_col}}} & \\multirow{{2}}{{*}}{{{ms_col}}}"
                else:
                    prefix = "& & "
                row_line = f"{prefix} & {mr} & " + " & ".join(row_cells) + r" \\"
                lines.append(re.sub(r" {2,}", " ", row_line))
            if defi < len(defenses) - 1:
                lines.append(rf"\cmidrule{{2-{last_col}}}")
        if di < len(datasets) - 1:
            lines.append(r"\midrule")
    return "\n".join(lines)


def render_both_defense_rows(idx, defenses, datasets=("mnist", "cifar10", "gtsrb"), mode="baseline", highlight_datasets=()):
    """Table 4 body rows only: both CAS and MS defenses applied together,
    neither side left at 'avg'. The paper's original both-defenses table has
    no MR column -- confirmed against the published numbers that it reports
    baseline (MR=no) only (e.g. cifar10/badnet/mkrum+mkrum: 5.43, matching
    our seed-0 baseline value exactly, not the replacement one).
    For datasets in `highlight_datasets`, each metric column in TABLE_METRICS
    is bolded (best) / red-boxed (worst) per attack across the full 16-row
    CAS x MS grid (ASR: lowest is best; Acc: highest is best -- see
    LOWER_IS_BETTER). No caption/label/tabular wrapper -- paste between
    \\toprule/\\bottomrule."""
    lines = []
    total_rows = len(defenses) * len(defenses)
    # last column index = data(1) + cas(1) + ms(1) + attacks*metrics (no MR column)
    last_col = 3 + len(ATTACKS) * len(TABLE_METRICS)
    for di, data in enumerate(datasets):
        dlabel = DATASET_LABEL.get(data, data)
        extrema = {
            metric: best_worst_metric_both(idx, data, defenses, mode, metric, LOWER_IS_BETTER[metric])
            for metric in TABLE_METRICS
        } if data in highlight_datasets else {}
        lines.append(rf"\multirow{{{total_rows}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{dlabel}}}}}")
        for msi, ms_defense in enumerate(defenses):
            ms_label = DEFENSE_LABEL.get(ms_defense, ms_defense)
            for casi, cas_defense in enumerate(defenses):
                cas_label = DEFENSE_LABEL.get(cas_defense, cas_defense)
                row_cells = []
                for attack in ATTACKS:
                    for metric in TABLE_METRICS:
                        cell = cell_from_idx(idx, data, attack, mode, cas_defense, ms_defense, metric)
                        if attack in extrema.get(metric, {}):
                            best_pair, worst_pair = extrema[metric][attack]
                            pair = (cas_defense, ms_defense)
                            if pair == best_pair:
                                cell = apply_highlight(cell, "bold")
                            elif pair == worst_pair and metric in RED_METRICS:
                                cell = apply_highlight(cell, "red")
                        row_cells.append(cell)
                ms_prefix = f"\\multirow{{{len(defenses)}}}{{*}}{{{ms_label}}}" if casi == 0 else ""
                row_line = f"& {cas_label} & {ms_prefix} & " + " & ".join(row_cells) + r" \\"
                lines.append(re.sub(r" {2,}", " ", row_line))
            if msi < len(defenses) - 1:
                lines.append(rf"\cmidrule{{2-{last_col}}}")
        if di < len(datasets) - 1:
            lines.append(r"\midrule")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shape 2: flat pivot, one row per (data,FS,MS,MR)
# ---------------------------------------------------------------------------

def build_by_ds_rows(rows):
    rows = [r for r in rows if r["attack"] != "CLEAN" and is_standard(r)]

    pivot = {}
    for r in rows:
        key = (r["data"], r["aggr"], r["aggr_server"], r["attack_mode"])
        pivot.setdefault(key, {})[r["attack"]] = r

    out_rows = []
    for (data, fs, ms, mode), by_attack in pivot.items():
        row = {
            "Dataset": DATASET_LABEL.get(data, data),
            "FS": fs, "MS": ms, "MR": MODE_TO_MR.get(mode, mode),
        }
        for attack in ATTACKS:
            label = ATTACK_COL_LABEL[attack]
            r = by_attack.get(attack)
            if r is None:
                row[f"{label}_ASR"] = "-"
                row[f"{label}_Acc"] = "-"
                row[f"{label}_RA"] = "-"
            else:
                row[f"{label}_ASR"] = cell_text(r, "asr")
                row[f"{label}_Acc"] = cell_text(r, "acc")
                row[f"{label}_RA"] = cell_text(r, "ra")
        out_rows.append(row)

    def sort_key(row):
        ds_idx = DATASET_ORDER.index(row["Dataset"]) if row["Dataset"] in DATASET_ORDER else 99
        mr_idx = ["no", "yes"].index(row["MR"]) if row["MR"] in ("no", "yes") else 99
        ms_idx = FS_MS_ORDER.index(row["MS"]) if row["MS"] in FS_MS_ORDER else 99
        fs_idx = FS_MS_ORDER.index(row["FS"]) if row["FS"] in FS_MS_ORDER else 99
        return (ds_idx, mr_idx, ms_idx, fs_idx)

    out_rows.sort(key=sort_key)
    return out_rows


def write_by_ds(out_rows):
    columns = ["Dataset", "FS", "MS", "MR"]
    for attack in ATTACKS:
        label = ATTACK_COL_LABEL[attack]
        columns += [f"{label}_ASR", f"{label}_Acc", f"{label}_RA"]

    out_csv = OUT_DIR / "seed_results_by_ds.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {out_csv.name}")

    out_txt = OUT_DIR / "seed_results_by_ds.txt"
    with out_txt.open("w") as f:
        f.write(" & ".join(columns) + r" \\" + "\n")
        for row in out_rows:
            f.write(" & ".join(str(row[c]) for c in columns) + r" \\" + "\n")
    print(f"Wrote {len(out_rows)} rows to {out_txt.name}")


def main():
    rows = load_rows()
    idx = index_rows(rows)
    defenses = ["mkrum", "foolsgold", "mmetric", "alignins", "feddlad"]
    both_defenses = ["mkrum", "alignins", "feddlad"]  # no mmetric, no foolsgold, no avg on either side

    files = {
        "table1_baseline_rows.tex": render_table1_rows(idx),
        "table2_CAS_defense_rows.tex": render_defense_table_rows(idx, defenses, role="CAS", highlight_datasets=("gtsrb",)),
        "table3_MS_defense_rows.tex": render_defense_table_rows(idx, defenses, role="MS", highlight_datasets=("gtsrb", "cifar10", "mnist")),
        "table4_both_defense_rows.tex": render_both_defense_rows(idx, both_defenses, highlight_datasets=("gtsrb", "cifar10", "mnist")),
    }
    for name, content in files.items():
        path = OUT_DIR / name
        path.write_text(content + "\n")
        print(f"Wrote {path.name}")

    write_by_ds(build_by_ds_rows(rows))


if __name__ == "__main__":
    main()
