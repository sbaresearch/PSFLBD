"""
Pull fresh run data from the wandb API across all 3 projects, keep only runs
that actually reached their target round count, group them by the full
experiment configuration (everything except seed), and report mean +/- std
across whichever of seed in {0, 43, 44} are available and complete.

Does NOT touch `current paper results.txt` -- this only reads config/summary
from wandb and writes fresh CSVs next to it for comparison.
"""
import csv
import statistics
from pathlib import Path

import wandb

PROJECTS = [
    "adavarts-tuwien/fl-backdoors-code_src_src_more",
    "adavarts-tuwien/split-fed-backdoor-3",
    "adavarts-tuwien/sfl2",
]

MIN_ROUNDS = 200

OUT_DIR = Path(__file__).parent
RAW_OUT = OUT_DIR / "seed_runs_raw.csv"
INCOMPLETE_OUT = OUT_DIR / "seed_runs_incomplete.csv"
AGG_OUT = OUT_DIR / "seed_results_aggregated.csv"
DUPES_OUT = OUT_DIR / "seed_runs_duplicates.csv"

AMBIGUOUS_OUT = OUT_DIR / "seed_runs_ambiguous_sfl2_seed0.csv"
PAPER_REF_PATH = OUT_DIR / "paper_reference_values.csv"

RAW_FIELDS = [
    "project", "run_id", "run_name", "state", "created_at",
    "round", "target_rounds", "complete",
    "seed", "seed_source", "data", "model", "attack", "attack_mode", "clean",
    "aggr", "aggr_server", "defense_start_round", "bg", "pg",
    "n_neighbors", "iqr_scale", "num_agents", "num_corrupt",
    "target_class", "poison_frac", "multiple_servers",
    "asr", "clean_acc", "robustness", "metrics_source",
]


def metrics_for_run(r, cfg, summ, complete):
    """Return (asr, clean_acc, robustness, metrics_source).

    For runs configured for exactly the standard 200-round comparison point
    (or fewer -- shouldn't happen given the completeness filter, but handled
    the same way), the final `run.summary` is the round-200 value and is
    cheap to use directly. But some runs were configured for MORE than 200
    rounds (`cfg.get("rounds", 200) > 200`); for those, `summary` reflects
    the FINAL round of training, not round 200, and is not comparable to the
    paper's "after 200 rounds" numbers. For those runs we instead pull
    `run.history()` and take the row at round==200 (falling back to the
    closest round <= 200, with a warning, if round==200 wasn't logged
    exactly).
    """
    rounds_cfg = cfg.get("rounds", MIN_ROUNDS)
    if not (complete and rounds_cfg and rounds_cfg > MIN_ROUNDS):
        return summ.get("asr"), summ.get("clean_acc"), summ.get("robustness acc"), "summary"

    try:
        hist = list(r.history(keys=["round", "asr", "clean_acc", "robustness acc"], pandas=False))
    except Exception as e:
        print(f"WARNING: history fetch failed for run {r.id} (rounds={rounds_cfg}): {e} "
              f"-- falling back to final summary (NOT round-200 comparable)")
        return summ.get("asr"), summ.get("clean_acc"), summ.get("robustness acc"), "summary_fallback_history_error"

    exact = [h for h in hist if h.get("round") == MIN_ROUNDS]
    if exact:
        row = exact[0]
        return row.get("asr"), row.get("clean_acc"), row.get("robustness acc"), "history_round200"

    le_200 = [h for h in hist if h.get("round") is not None and h.get("round") <= MIN_ROUNDS]
    if le_200:
        row = max(le_200, key=lambda h: h["round"])
        print(f"WARNING: run {r.id} (rounds={rounds_cfg}) has no exact round==200 history row; "
              f"falling back to closest round<={MIN_ROUNDS} found (round={row.get('round')})")
        return row.get("asr"), row.get("clean_acc"), row.get("robustness acc"), "history_closest_le_200"

    print(f"WARNING: run {r.id} (rounds={rounds_cfg}) has no history row with round<={MIN_ROUNDS}; "
          f"falling back to final summary (NOT round-200 comparable)")
    return summ.get("asr"), summ.get("clean_acc"), summ.get("robustness acc"), "summary_fallback_no_history_le_200"


def fetch_all_runs():
    records = []
    for project in PROJECTS:
        runs = wandb.Api().runs(project, per_page=200)
        print(f"Fetched {len(runs)} runs from {project}")
        for r in runs:
            cfg = r.config
            summ = dict(r.summary)
            round_val = summ.get("round")
            target_rounds = cfg.get("rounds") or MIN_ROUNDS
            target_rounds = max(target_rounds, MIN_ROUNDS)
            complete = round_val is not None and round_val >= target_rounds
            seed_source = "legacy_implicit0" if "seed" not in cfg else "explicit"
            asr_val, clean_acc_val, robustness_val, metrics_source = metrics_for_run(r, cfg, summ, complete)

            records.append({
                "project": project,
                "run_id": r.id,
                "run_name": r.name,
                "state": r.state,
                "created_at": r.created_at,
                "round": round_val,
                "target_rounds": target_rounds,
                "complete": complete,
                "seed": cfg.get("seed", 0),
                "seed_source": seed_source,
                "data": cfg.get("data"),
                "model": cfg.get("model", "resnet18"),
                "attack": cfg.get("attack"),
                "attack_mode": cfg.get("attack_mode"),
                "clean": bool(cfg.get("clean", False)),
                "aggr": cfg.get("aggr"),
                "aggr_server": cfg.get("aggr_server"),
                "defense_start_round": cfg.get("defense_start_round", 1),
                "bg": cfg.get("bg"),
                "pg": cfg.get("pg"),
                "n_neighbors": cfg.get("n_neighbors"),
                "iqr_scale": cfg.get("iqr_scale"),
                "num_agents": cfg.get("num_agents"),
                "num_corrupt": cfg.get("num_corrupt"),
                "target_class": cfg.get("target_class"),
                "poison_frac": cfg.get("poison_frac"),
                "multiple_servers": cfg.get("multiple_servers", False),
                "asr": asr_val,
                "clean_acc": clean_acc_val,
                "robustness": robustness_val,
                "metrics_source": metrics_source,
            })
    return records


def group_key(rec):
    if rec["clean"]:
        attack = "CLEAN"
        attack_mode = None
    else:
        attack = rec["attack"]
        attack_mode = rec["attack_mode"]
    # multiple_servers is documented in sfl.py's sfl_variant docstring as a
    # memory-optimization-only flag ("same math") for the default v1 variant.
    # Originally only merged for GTSRB (2026-08-04 decision, where the paper's
    # True-logged baseline runs needed to pair with the False-logged seed-43/44
    # sweep). Extended to all datasets (2026-08-13): a stray cifar10 seed-44 run
    # surfaced with multiple_servers=True against an otherwise-False group,
    # and the user confirmed the flag should just be ignored for grouping
    # everywhere, not only GTSRB.
    multiple_servers = "merged_ms_true_false"
    # target_class: the paper's GTSRB attack runs used target_class=12
    # (sweeps/base/BadNet_GTSRB_all.yaml); the current seed-43/44 sweeps use
    # target_class=7 (attacks_GTSRB_all.yaml etc). Confirmed with the user
    # (2026-08-04) this discrepancy doesn't matter -- merge it for GTSRB, same
    # treatment as multiple_servers above.
    target_class = "merged_target_class" if rec["data"] == "gtsrb" else rec["target_class"]
    return (
        rec["data"], rec["model"], attack, attack_mode,
        rec["aggr"], rec["aggr_server"], rec["defense_start_round"],
        rec["bg"], rec["pg"], rec["n_neighbors"], rec["iqr_scale"],
        rec["num_agents"], rec["num_corrupt"], target_class,
        rec["poison_frac"], multiple_servers,
    )


GROUP_LABELS = [
    "data", "model", "attack", "attack_mode",
    "aggr", "aggr_server", "defense_start_round",
    "bg", "pg", "n_neighbors", "iqr_scale",
    "num_agents", "num_corrupt", "target_class",
    "poison_frac", "multiple_servers",
]


def mean_std(values):
    if len(values) == 1:
        return values[0], None
    return statistics.mean(values), statistics.stdev(values)


def ref_subkey(full_key):
    """The paper-reference lookup key is a subset of the full group key:
    (data, model, attack-or-CLEAN, attack_mode-or-None, aggr, aggr_server)."""
    return tuple(full_key[:6])


def load_paper_reference():
    """Load `paper_reference_values.csv` (hand-transcribed from the published
    LaTeX, read-only) into {ref_subkey: {asr_pct, acc_pct, ra_pct}}. Returns
    {} if the file doesn't exist (keeps the script runnable without it)."""
    ref = {}
    if not PAPER_REF_PATH.exists():
        print(f"WARNING: {PAPER_REF_PATH.name} not found -- skipping paper-reference "
              f"GTSRB seed0 promotion, all sfl2/seed0 GTSRB runs stay excluded.")
        return ref
    with PAPER_REF_PATH.open() as f:
        for row in csv.DictReader(f):
            attack = row["attack"] or None
            attack_mode = row["attack_mode"] or None
            if attack == "CLEAN":
                attack_mode = None
            key = (row["dataset"], row["model"], attack, attack_mode, row["aggr"], row["aggr_server"])
            ref[key] = {
                "asr_pct": float(row["asr_pct"]) if row["asr_pct"] else None,
                "acc_pct": float(row["acc_pct"]) if row["acc_pct"] else None,
                "ra_pct": float(row["ra_pct"]) if row["ra_pct"] else None,
            }
    return ref


def matches_reference(rec, ref_vals, tol=0.01):
    """True if rec's asr/clean_acc/robustness (as pct) match every non-blank
    field of ref_vals within `tol` absolute percentage points."""
    checks = [
        (ref_vals["asr_pct"], rec["asr"]),
        (ref_vals["acc_pct"], rec["clean_acc"]),
        (ref_vals["ra_pct"], rec["robustness"]),
    ]
    for ref_pct, val in checks:
        if ref_pct is None:
            continue
        if val is None:
            return False
        if abs(round(val * 100, 2) - ref_pct) > tol:
            return False
    return True


def promote_gtsrb_seed0(groups, ambiguous, paper_ref):
    """Decision 2: don't blanket-exclude sfl2/seed==0 GTSRB runs. For every
    GTSRB group (existing in `groups`, or only present in the ambiguous pool),
    try to promote one sfl2/seed==0 candidate as that group's seed=0 value:

      - reference row exists AND a candidate matches it (within tolerance)
        -> promote most-recently-created matching candidate, tag exact_paper_match
      - reference row exists but NO candidate matches it
        -> promote most-recently-created candidate anyway (best effort), tag
           reference_exists_no_match (so the user can see it did NOT reproduce
           the published number)
      - no reference row at all for this config
        -> promote most-recently-created candidate, tag recency_fallback_no_reference

    Mutates `groups` in place (adds seed 0 entries / new group keys as needed).
    Returns {group_key: provenance_str} for GTSRB groups that got a seed0.
    """
    ambiguous_by_key = {}
    for rec in ambiguous:
        ambiguous_by_key.setdefault(group_key(rec), []).append(rec)

    gtsrb_keys = {k for k in groups if k[0] == "gtsrb"}
    gtsrb_keys |= {k for k in ambiguous_by_key if k[0] == "gtsrb"}

    provenance_by_key = {}
    for key in gtsrb_keys:
        candidates = ambiguous_by_key.get(key, [])
        ref_vals = paper_ref.get(ref_subkey(key))

        promoted, provenance = None, ""
        if ref_vals is not None:
            matches = [c for c in candidates if matches_reference(c, ref_vals)]
            if matches:
                promoted = sorted(matches, key=lambda r: r["created_at"])[-1]
                provenance = "exact_paper_match"
            elif candidates:
                promoted = sorted(candidates, key=lambda r: r["created_at"])[-1]
                provenance = "reference_exists_no_match"
            else:
                provenance = "reference_exists_no_match"
        elif candidates:
            promoted = sorted(candidates, key=lambda r: r["created_at"])[-1]
            provenance = "recency_fallback_no_reference"

        if promoted is not None:
            groups.setdefault(key, {})[0] = [promoted]
            provenance_by_key[key] = provenance
        elif provenance:
            # reference exists, but literally zero sfl2/seed0 candidates for
            # this exact config -- nothing to promote, but still worth a tag
            # if the group already exists in `groups` via 43/44.
            if key in groups:
                provenance_by_key[key] = provenance

    return provenance_by_key


def main():
    records = fetch_all_runs()

    complete = [r for r in records if r["complete"]]
    incomplete = [r for r in records if not r["complete"]]

    print(f"\n{len(complete)} complete runs (>= {MIN_ROUNDS} rounds), "
          f"{len(incomplete)} incomplete/in-progress runs excluded")

    with RAW_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        w.writerows(records)
    print(f"Wrote all {len(records)} runs (raw) to {RAW_OUT.name}")

    with INCOMPLETE_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        w.writerows(incomplete)
    print(f"Wrote {len(incomplete)} incomplete/excluded runs to {INCOMPLETE_OUT.name}")

    # sfl2 has its own seed==0 runs that are leftover dev/hyperparameter churn
    # predating the formal seed-43/44 sweeps (near-duplicate configs with wildly
    # different metrics). NOTE: on inspection these turn out to have NO `seed`
    # key at all (seed_source=="legacy_implicit0"), same as the two truly-legacy
    # projects -- they predate the seed feature within sfl2 itself, created
    # 2026-06-17..06-27, before the explicit seed=43/44 sweep configs existed.
    # So seed_source alone can't tell them apart from genuine legacy-project
    # runs; we key the exclusion off *project identity* instead: any sfl2 run
    # that resolves to seed==0 (explicit or missing-key) is dev churn, NOT the
    # paper's seed-0 source (that's the two dedicated legacy no-`seed`-key
    # projects) -- exclude from aggregation/grouping, save for manual review.
    # NOTE (decision 2 update): for GTSRB specifically, some of these ARE the
    # paper's real seed-0 source (the two legacy projects have zero GTSRB
    # runs at all) -- see promote_gtsrb_seed0() below, which re-admits the
    # ones that reproduce (or best-effort approximate) the published numbers.
    # For CIFAR-10/MNIST the legacy projects remain the trusted seed-0 source
    # and this exclusion stands unmodified.
    ambiguous = [r for r in complete if r["project"] == "adavarts-tuwien/sfl2" and r["seed"] == 0]
    complete_for_agg = [r for r in complete if not (r["project"] == "adavarts-tuwien/sfl2" and r["seed"] == 0)]
    print(f"\n{len(ambiguous)} runs excluded as ambiguous: seed==0 runs from sfl2 project itself "
          f"(likely dev/hyperparameter-tuning churn predating the seed-43/44 sweeps; "
          f"GTSRB candidates among these get re-evaluated below -- see {AMBIGUOUS_OUT.name} "
          f"for the full unfiltered set)")
    if ambiguous:
        with AMBIGUOUS_OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
            w.writeheader()
            w.writerows(ambiguous)

    # group complete runs (excluding ambiguous sfl2 seed==0) by full config
    # (minus seed), then by seed within group
    groups = {}
    for rec in complete_for_agg:
        key = group_key(rec)
        groups.setdefault(key, {}).setdefault(rec["seed"], []).append(rec)

    # Decision 2: GTSRB really is executed inside sfl2, and the two "legacy"
    # projects have zero GTSRB runs at all -- blanket-excluding sfl2/seed==0
    # would leave every GTSRB group with no seed0 whatsoever. Instead,
    # cross-check each GTSRB group's sfl2/seed==0 candidates against the
    # paper's published numbers (paper_reference_values.csv) and promote the
    # best candidate back in, tagged with how confident that promotion is.
    paper_ref = load_paper_reference()
    provenance_by_key = promote_gtsrb_seed0(groups, ambiguous, paper_ref)
    n_exact = sum(1 for v in provenance_by_key.values() if v == "exact_paper_match")
    n_no_match = sum(1 for v in provenance_by_key.values() if v == "reference_exists_no_match")
    n_fallback = sum(1 for v in provenance_by_key.values() if v == "recency_fallback_no_reference")
    print(f"\nGTSRB seed0 promotion: {n_exact} exact_paper_match, "
          f"{n_no_match} reference_exists_no_match, {n_fallback} recency_fallback_no_reference "
          f"({len(provenance_by_key)} GTSRB groups total got a seed0_provenance tag)")

    dupes = []
    agg_rows = []
    for key, by_seed in groups.items():
        seed_values = {}  # seed -> chosen rec
        for seed, recs in by_seed.items():
            if len(recs) > 1:
                dupes.extend(recs)
                # pick the most recently created run as the authoritative one
                recs = sorted(recs, key=lambda r: r["created_at"])
            seed_values[seed] = recs[-1]

        seeds_present = sorted(seed_values.keys())
        asr_vals = [seed_values[s]["asr"] for s in seeds_present if seed_values[s]["asr"] is not None]
        acc_vals = [seed_values[s]["clean_acc"] for s in seeds_present if seed_values[s]["clean_acc"] is not None]
        ra_vals = [seed_values[s]["robustness"] for s in seeds_present if seed_values[s]["robustness"] is not None]

        asr_mean, asr_std = mean_std(asr_vals) if asr_vals else (None, None)
        acc_mean, acc_std = mean_std(acc_vals) if acc_vals else (None, None)
        ra_mean, ra_std = mean_std(ra_vals) if ra_vals else (None, None)

        row = dict(zip(GROUP_LABELS, key))
        row["seeds_present"] = ",".join(str(s) for s in seeds_present)
        row["n_seeds"] = len(seeds_present)
        row["asr_mean_pct"] = f"{asr_mean*100:.2f}" if asr_mean is not None else ""
        row["asr_std_pct"] = f"{asr_std*100:.2f}" if asr_std is not None else ""
        row["acc_mean_pct"] = f"{acc_mean*100:.2f}" if acc_mean is not None else ""
        row["acc_std_pct"] = f"{acc_std*100:.2f}" if acc_std is not None else ""
        row["ra_mean_pct"] = f"{ra_mean*100:.2f}" if ra_mean is not None else ""
        row["ra_std_pct"] = f"{ra_std*100:.2f}" if ra_std is not None else ""
        for s in (0, 43, 44):
            r = seed_values.get(s)
            row[f"seed{s}_asr_pct"] = f"{r['asr']*100:.2f}" if r and r["asr"] is not None else ""
            row[f"seed{s}_acc_pct"] = f"{r['clean_acc']*100:.2f}" if r and r["clean_acc"] is not None else ""
            row[f"seed{s}_ra_pct"] = f"{r['robustness']*100:.2f}" if r and r["robustness"] is not None else ""
        row["seed0_provenance"] = provenance_by_key.get(key, "")
        agg_rows.append(row)

    fieldnames = GROUP_LABELS + [
        "seeds_present", "n_seeds",
        "asr_mean_pct", "asr_std_pct",
        "acc_mean_pct", "acc_std_pct",
        "ra_mean_pct", "ra_std_pct",
        "seed0_asr_pct", "seed0_acc_pct", "seed0_ra_pct", "seed0_provenance",
        "seed43_asr_pct", "seed43_acc_pct", "seed43_ra_pct",
        "seed44_asr_pct", "seed44_acc_pct", "seed44_ra_pct",
    ]
    agg_rows.sort(key=lambda r: tuple(str(r[k]) for k in GROUP_LABELS))
    with AGG_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(agg_rows)
    print(f"Wrote {len(agg_rows)} aggregated (config -> mean+-std over seeds) rows to {AGG_OUT.name}")

    if dupes:
        with DUPES_OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
            w.writeheader()
            w.writerows(dupes)
        print(f"WARNING: {len(dupes)} runs were duplicates within the same (config, seed) -- "
              f"see {DUPES_OUT.name}; kept the most recently created one per group.")


if __name__ == "__main__":
    main()
