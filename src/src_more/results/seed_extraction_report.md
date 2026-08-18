# Seed Extraction Report (wandb API, generated 2026-08-04, updated same day after coordinator review)

## Addendum: Decisions 3 & 4 (same day, after further review)

Two more grouping-key fields turned out to be spurious blockers for merging GTSRB's
seed-0 data with the live seed-43/44 sweeps, and were resolved with the user directly:

**Decision 3 — `multiple_servers`.** Documented in `sfl.py`'s `sfl_variant` docstring as a
memory-optimization-only flag ("same math") for the default v1 variant. Empirical check
(only one same-config pair existed, cifar10 clean seed 0: `multiple_servers=False` gave
61.73% vs `multiple_servers=True` gave 43.06%, an 18.7pp gap) could not confirm this from
data alone, and no GTSRB pair existed to check directly. Cross-check: GTSRB's own
seed-to-seed noise floor (seed 43 vs 44, same config, both `multiple_servers=False`) is
already 10-32pp on ASR/RA — comparable to or larger than that 18.7pp gap — so a real
`multiple_servers` effect (if any) would be indistinguishable from ordinary seed noise for
GTSRB anyway. All CIFAR-10 runs are `multiple_servers=False` except one stray dev run, so
this is a non-issue there. **User decision: merge `multiple_servers` for GTSRB only** (kept
as a distinct grouping field for cifar10/mnist, where it doesn't matter).

**Decision 4 — `target_class`.** Applying Decision 3 alone did *not* actually merge the
GTSRB Table-1 groups as expected — investigating why revealed the real blocker: all 6 of
the exact-matching Table-1 attack runs (badnet/DBA/neurotoxin x baseline/replacement) use
`target_class=12` (from `sweeps/base/BadNet_GTSRB_all.yaml`), while every current
seed-43/44 sweep config uses `target_class=7` (`attacks_GTSRB_all.yaml`,
`attacks_GTSRB_all_repl.yaml`, `FS_GT.yaml`). This is a real backdoor-target parameter, not
a memory/implementation detail like Decision 3 — flagged to the user as time-sensitive
since the seed-43/44 GTSRB sweeps were actively running against `target_class=7` while the
paper's number came from `target_class=12`. **User decision: target_class is fine, merge
it too** (same technique as Decision 3 — normalized out of the GTSRB grouping key only;
cifar10/mnist unaffected, they're uniformly `target_class=7` throughout).

### Result after both merges

- GTSRB seed0 promotion: **65 exact_paper_match, 2 reference_exists_no_match** (down from
  13 before Decision 3/4 — the other 11 were exactly the `multiple_servers`/`target_class`
  mismatch cases now resolved), **79 recency_fallback_no_reference**. 146 of 156 GTSRB
  groups have a seed0 value (up from 0 before Decision 2, unchanged count from Decision
  2 -- Decisions 3/4 didn't add *new* seed0 values, they let the *existing* ones merge
  with seed43/44 into the same group).
- **GTSRB now has 11 groups with 2+ seeds** (9 with the full `0,43,44`, 1 with `0,43` only,
  1 with `0,44` only) — up from 0 before this addendum. Total across all datasets: **29
  groups with 2+ seeds** (up from 27).
- The 2 remaining `reference_exists_no_match` cases are the same genuine dev-churn cases
  identified earlier (§2b below): mmetric-CAS/badnet/replacement and
  feddlad-CAS/neurotoxin/replacement, where no single candidate run reproduces the full
  published (ASR, Acc, RA) triple. Not resolved by Decisions 3/4 — these are real
  unverifiable cases, not a grouping-key artifact.
- Sanity check on the merged GTSRB 3-seed groups — std is large for several (e.g.
  DBA/mkrum/avg: ASR 67.58% ± 20.14pp; badnet/baseline/avg/avg: ASR 83.01% ± 15.02pp).
  This is consistent with, not caused by, the merge: GTSRB's inherent seed-to-seed
  variance was already shown (Decision 3 investigation) to be 10-32pp even *within*
  `multiple_servers=False` alone — these stds reflect that real volatility, not a
  measurement error from merging non-equivalent configs.

Everything below this addendum reflects the *prior* state (before Decisions 3/4) and is
kept for the historical record of how the investigation proceeded; §1 (run counts) and the
paper cross-check numbers (§3) are unaffected by Decisions 3/4 — only §2 (GTSRB promotion
counts) and §4 (coverage) changed as described above.

Pipeline: `extract_seed_results.py`. Outputs used below: `seed_runs_raw.csv`,
`seed_runs_incomplete.csv`, `seed_runs_ambiguous_sfl2_seed0.csv`,
`seed_runs_duplicates.csv`, `seed_results_aggregated.csv`,
`paper_reference_values.csv` (read-only, hand-transcribed from the paper).

**Two decisions were applied after the first pass and are reflected throughout this version:**
1. **Round-200-vs-final-summary fix** — any complete run configured for `rounds > 200` now has its asr/clean_acc/robustness pulled from `run.history()` at `round==200` instead of the final `run.summary` (which reflected whatever round the run actually stopped at). Only 7 of 511 runs needed this.
2. **GTSRB seed-0 promotion** — sfl2's own seed==0 runs are no longer blanket-excluded for GTSRB. Each GTSRB group's sfl2/seed==0 candidates are checked against `paper_reference_values.csv`; matches are promoted back in with a provenance tag. CIFAR-10/MNIST are untouched by this (still legacy-projects-only for seed 0).

## 1. Run counts

**511 runs fetched total**, from 3 wandb projects (one more sfl2 run than the prior pass — a live sweep run either finished or a new one appeared between the two runs of the script).

| Project | Total | Complete (>=200 rounds) | Incomplete/excluded |
|---|---|---|---|
| `fl-backdoors-code_src_src_more` | 245 | 220 | 25 |
| `split-fed-backdoor-3` | 24 | 20 | 4 |
| `sfl2` | 242 | 221 | 21 |
| **Total** | **511** | **461** | **50** |

By dataset:

| Dataset | Total | Complete | Incomplete |
|---|---|---|---|
| cifar10 | 191 | 157 | 34 |
| gtsrb | 192 | 182 | 10 |
| mnist | 122 | 121 | 1 |
| femnist | 4 | 0 | 4 |
| ets | 1 | 1 | 0 |

**Why the 50 incomplete runs were excluded** (none silently dropped, all in `seed_runs_incomplete.csv`):

- **25 cifar10 runs, `fl-backdoors-code_src_src_more`, state=`failed`**: no `round` in summary at all (never logged anything usable) — all `feddlad`/`alignins`/`foolsgold`/`mkrum` combo configs, seed 0.
- **4 femnist runs, `split-fed-backdoor-3`, state=`killed`**: stopped at round 4/28/20/132 of 200 — femnist has zero complete runs anywhere.
- **1 sfl2 run** (`n05pk0mv`, `denim-forest-208`): finished but config/summary essentially empty (no `data`, no `round`) — looks like a crashed-before-init sweep launch.
- **2 sfl2 gtsrb runs, state=`failed`**: `feddlad` defense, died at round 164/200 and 176/200.
- **13 sfl2 runs, state=`running`** (5 cifar10 + 8 gtsrb): currently live sweep runs, at rounds ranging from 5/200 to 195/200 (e.g. `imu1ddi1` cifar10 badnet baseline seed44 at 100/200; `yt1nw332` gtsrb badnet+alignins seed44 at 195/200). These are the active seed-43/44 sweeps the user mentioned — will complete on their own; **left completely untouched**, no GPU/process access of any kind.
- **5 sfl2 runs, state=`crashed`**: died mid-run at rounds 144–169/200 (cifar10 seed43/44 x3, mnist seed43 x1).

None of these were touched, killed, or restarted — purely read via the wandb API and filed to `seed_runs_incomplete.csv` for a future re-run.

**Metrics-source breakdown** (Decision 1): 504 of 511 runs used `run.summary` directly (cheap path, `cfg.get("rounds",200) <= 200`); **7 runs** (all configured for `rounds=500`) used `run.history()` at `round==200` instead — all 7 had an exact `round==200` row logged, so no closest-round fallback was needed. The 7: `a90i3d29` (cifar10 clean, legacy project) plus 6 GTSRB "both-defenses" runs (`kh37871r`, `gjj03htw`, `l6b6qmn6`, `pifxcm6l`, `8y8apgrr`, `fle72u4p` — alignins/alignins and mkrum/mkrum combos, all sfl2/seed0/ambiguous pool).

## 2. Seed-0 ambiguity: finding and decisions

**Original premise (from task setup) needed correcting.** The setup described sfl2's ~164 "seed==0" runs as having an *explicit* `config['seed'] == 0`. Direct inspection of the raw wandb config (`'seed' in r.config`) shows this is wrong: those runs have **no `seed` key at all** — same as the two legacy projects — because they predate the `seed` feature *within sfl2 itself* (created 2026-06-16 to 2026-06-27, before the explicit seed=43/44 sweep configs existed, which only start appearing 2026-08-02/03/04). The exclusion is keyed off **project identity** instead: any `sfl2` run resolving to seed==0 is filed as "ambiguous." Result: **164 runs**, written unfiltered (all 164, no shrinking) to `seed_runs_ambiguous_sfl2_seed0.csv`.

**Decision 1 was reached after finding the two legacy projects have zero GTSRB runs of any kind** — GTSRB only exists in sfl2. Blanket-excluding sfl2/seed==0 left every GTSRB group with no seed0 at all in the aggregated CSV, even though several excluded runs reproduced the paper's numbers exactly. The coordinator/user reviewed this and decided: **don't exclude GTSRB wholesale — promote the runs that are verifiably correct, and be explicit about which ones are best-effort guesses.**

### How promotion works

For each GTSRB experiment-config group (matched on data, model, attack-or-CLEAN, attack_mode, aggr, aggr_server — the same lookup key as `paper_reference_values.csv`), every sfl2/seed==0 candidate in that group is checked against the published reference (tolerance 0.01 pp on each of asr/acc/ra that the paper reports):

| Outcome | Meaning | Count |
|---|---|---|
| `exact_paper_match` | A candidate reproduces every published metric for this config (within 0.01pp) → promoted | **65** |
| `reference_exists_no_match` | The paper published a value for this config, but no candidate reproduces it → best-effort candidate promoted anyway (or none, if zero candidates exist), tagged so it's visibly unverified | **13** |
| `recency_fallback_no_reference` | No published reference exists for this config (Table 4 both-defenses combos, non-default configs, etc.) → most-recently-created candidate promoted as a best-effort seed0 | **79** |
| **Total GTSRB groups touched** | | **157** |

146 of these 157 groups ended up with an actual seed0 number (11 of the 13 `reference_exists_no_match` cases had **zero** candidates at all for that exact grouping — see below — so nothing could be promoted there).

### The 13 `reference_exists_no_match` cases, in full

Two distinct situations produced this tag — worth separating clearly rather than treating as one bucket:

**(a) 11 cases: a published reference exists, but it belongs to a *different*, `multiple_servers=True` grouping, and this exact `multiple_servers=False` grouping (the one aligned with the current seed-43/44 sweep) has zero sfl2/seed0 candidates at all.** This is not a "wrong answer" — it's simply "no seed-0 data exists yet for this specific config variant." Example: GTSRB badnet/baseline/avg/avg has **three** distinct groups in the data — `multiple_servers=True` (seed0 only, exact match 96.89/80.54/2.84, run `0u2zj300`), `multiple_servers=False` seed 43/44 (85.09/78.32/11.29 and 67.06/79.85/23.96 — no seed0 candidate exists for this variant), and a `model=resnet18` variant (seed0 only, `recency_fallback_no_reference`, since the paper reference is googlenet-only). **This is a genuinely useful finding for the user independent of the extraction pipeline: the paper's baseline table was built with `multiple_servers=True`, but the currently-running seed-43/44 sweep uses `multiple_servers=False` — those are two different experimental setups and will never merge into one seed0/43/44 mean±std until that's reconciled.** Affected configs (all `multiple_servers=False`, `aggr_server=avg` unless noted): CLEAN avg/avg; DBA baseline+replacement avg/avg and mkrum/avg; badnet baseline+replacement avg/avg, mkrum/avg (x2 modes), alignins/avg (44 only); neurotoxin baseline mkrum/avg.

**(b) 2 cases: a candidate DOES exist, but it doesn't reproduce the published triple (asr, acc, ra) simultaneously** — genuine unverifiable cases, flagged loudly as requested:

| Config | Published (ASR/Acc/RA) | Best candidate promoted | Run ID | What else was tried |
|---|---|---|---|---|
| gtsrb, mmetric-as-CAS, badnet, replacement | 76.57 / 81.28 / 24.93 | 76.57 / 80.09 / 18.69 (ASR matches, Acc/RA don't) | `3xe53d03` (created 2026-06-26 13:55) | `0sp2snsx` (2026-06-26 02:19): 68.12/81.27/24.92 — Acc/RA match almost exactly, ASR doesn't. `qqclo2xm` (2026-06-19, mode=baseline): identical to `3xe53d03`'s numbers. No single run reproduces all three published metrics — real dev-churn split across ≥3 runs for this config. |
| gtsrb, feddlad-as-CAS, neurotoxin, replacement | 96.93 / 81.58 / 2.74 | 98.55 / 82.75 / 1.16 (close, not exact) | `p710cgrl` (2026-06-25 19:00) | 7 other `baseline`-mode feddlad/neurotoxin candidates exist from 2026-06-19 to 2026-06-26 with ASR ranging **0.17% to 96.9%** for the nominally same config — heavy dev-churn on this specific defense combo. `replacement` mode only has this one candidate, and it's close-but-not-exact. |

Both are exactly the kind of "wildly swinging duplicate" dev-churn the original task setup warned about — it's real, but concentrated in a handful of defended (CAS/MS) GTSRB configs, not in the plain Table-1 baseline rows, which are clean singletons (see §3).

## 3. Paper cross-check

Extracted values: `run.summary[...]` for runs with `rounds<=200` (504/511), else `run.history()` at `round==200` (7/511, per Decision 1). "Source" column: `agg (seed0, legacy)` = cifar10/mnist legacy-project seed0; `agg (seed0, promoted: <tag>)` = GTSRB value now sitting in `seed_results_aggregated.csv` via the promotion logic in §2.

### Clean accuracy after 200 rounds — Decision 1 resolved

| Dataset | Published | Extracted (now) | Run ID | Source | Match |
|---|---|---|---|---|---|
| cifar10 | 61.93% | **61.73%** (history @ round 200) | `a90i3d29` (split-fed-backdoor-3) | agg (seed0, legacy) | Close (0.20pp off) — accepted as typo in paper text, not a pipeline bug |
| gtsrb | 82.28% | 82.28% | `x8xg6ak4` (sfl2, seed 0) | agg (seed0, promoted: exact_paper_match) | Exact match |

Before the fix, `run.summary` gave 76.18% for cifar10 (`a90i3d29` was configured for **500 rounds**, so summary reflected the final round, not round 200). Per the coordinator/user's call, the residual 61.73% vs. 61.93% gap (0.20pp) is treated as most likely a digit-transposition typo in the paper text ("93" vs "73") rather than a pipeline issue, and **61.73% is now the accepted seed-0 CIFAR-10 clean value** going forward — no further chasing planned.

### Table 1 — Baseline attacks without defenses (aggr=avg, aggr_server=avg)

| Dataset | Attack | Mode | Published (ASR/Acc/RA) | Extracted (ASR/Acc/RA) | Run ID | Source | Match |
|---|---|---|---|---|---|---|---|
| cifar10 | badnet | baseline | 89.62/60.21/7.23 | 89.62/60.21/7.23 | `opgam996` | agg (seed0, legacy) | Exact |
| cifar10 | badnet | replacement | 89.62/60.21/7.23 | 89.62/60.21/7.23 | `iciqwh4b` | agg (seed0, legacy) | Exact |
| cifar10 | DBA | baseline | 15.58/58.97/52.88 | 15.58/58.97/52.88 | `2nt6cziw` | agg (seed0, legacy) | Exact |
| cifar10 | DBA | replacement | 15.58/58.97/52.88 | 15.58/58.97/52.88 | `bw4m0fsv` | agg (seed0, legacy) | Exact |
| cifar10 | neurotoxin | baseline | 90.69/60.58/6.51 | 90.69/60.58/6.51 | `lnrsu1ck` | agg (seed0, legacy) | Exact |
| cifar10 | neurotoxin | replacement | 90.77/60.59/6.47 | 90.77/60.59/6.47 | `6orbdida` | agg (seed0, legacy) | Exact |
| gtsrb (multiple_servers=True) | badnet | baseline/repl. (dup. row) | 96.89/80.54/2.84 | 96.89/80.54/2.84 | `0u2zj300` (baseline), `9ry2rrs1` (repl.) | agg (seed0, promoted: exact_paper_match) | Exact |
| gtsrb (multiple_servers=True) | DBA | baseline/repl. | 48.63/79.86/40.70 | 48.63/79.86/40.70 | `de0ocd5n` (baseline), `85ij4tzk` (repl.) | agg (seed0, promoted: exact_paper_match) | Exact |
| gtsrb (multiple_servers=True) | neurotoxin | baseline/repl. | 99.39/82.22/0.59 | 99.39/82.22/0.59 | `4gceeas3` (baseline), `n3ht92yg` (repl.) | agg (seed0, promoted: exact_paper_match) | Exact |

All 9 Table‑1 rows match the published table exactly to 2 decimal places, and **all 9 GTSRB rows are now present in `seed_results_aggregated.csv`** (previously they only existed in the excluded ambiguous file). Note: these GTSRB Table-1 matches all have `multiple_servers=True` — the *separate* `multiple_servers=False` groupings (which is what the live seed-43/44 sweep is using) don't have a seed0 counterpart at all yet (see §2a).

### Table 2 / 3 spot checks (CIFAR-10, MKrum)

| Config | Attack | Published (ASR/Acc/RA) | Extracted (ASR/Acc/RA) | Run ID | Match |
|---|---|---|---|---|---|
| MKrum-as-CAS (aggr=mkrum, aggr_server=avg), baseline | badnet | 89.52/60.25/7.36 | 89.52/60.25/7.36 | `lucvjtwd` | Exact |
| MKrum-as-CAS, baseline | DBA | 15.71/59.11/52.76 | 15.71/59.11/52.76 | `lllty374` | Exact |
| MKrum-as-CAS, baseline | neurotoxin | 90.59/60.51/6.58 | 90.59/60.51/6.58 | `2g5wm5mv` | Exact |
| MKrum-as-MS (aggr=avg, aggr_server=mkrum), baseline | badnet | 5.54/62.00/59.04 | 5.54/62.00/59.04 | `hv7sihqx` | Exact |
| MKrum-as-MS, baseline | DBA | 5.88/61.99/58.63 | 5.88/61.99/58.63 | `enqokn2f` | Exact |
| MKrum-as-MS, baseline | neurotoxin | 5.88/61.88/58.94 | 5.88/61.88/58.94 | `qf6a59pj` | Exact |

All 6 spot-checked rows match exactly (unaffected by either decision — no cifar10 run here exceeds 200 rounds, and this section doesn't touch the GTSRB promotion logic). One note preserved from the first pass: for MKrum-as-MS/badnet/baseline there are two seed-0 candidates with the same nominal label but genuinely different configs (`5d7xd440`: num_agents=10/num_corrupt=2/poison_frac=0.3, a different non-default experiment vs. `hv7sihqx`: the standard 20/4/0.5 config matching the paper) — correctly placed in different groups, not a bug.

**Cross-check summary: 15 of 16 explicitly spot-checked rows match the paper exactly.** The one soft mismatch (CIFAR-10 clean accuracy, 0.20pp) is accepted as a likely paper typo per the coordinator's decision. Separately, §2 surfaced 2 additional GTSRB defended-config rows (mmetric-CAS/badnet/replacement, feddlad-CAS/neurotoxin/replacement) where no single run reproduces the published triple — flagged as genuinely unverified, not smoothed over.

## 4. Coverage summary (seed availability per experiment group)

**414 total groups** in `seed_results_aggregated.csv` (up from 267 before Decision 2 — GTSRB gained 147 new groups that previously had no seed0 entry point at all). Seed presence distribution:

| Seeds present | # groups | Meaning |
|---|---|---|
| `0,43,44` | 1 | Full 3-seed mean±std (cifar10 CLEAN/avg/avg — now a clean, consistent std after the round-200 fix, see below) |
| `43,44` | 26 | 2-seed mean±std from the new sweeps; GTSRB variants here are all `multiple_servers=False`, with no promoted seed0 counterpart (§2a) |
| `44` only / `43` only | 2 | Sweep partially through |
| `0` only | 384 | Single value pending the sweeps reaching that config (238 legacy cifar10/mnist + 146 GTSRB, the latter now populated thanks to Decision 2) |

By dataset:

| Dataset | `0` only | `43,44` | `0,43,44` | `43`/`44` only |
|---|---|---|---|---|
| cifar10 | 119 | 16 | 1 | 1 |
| mnist | 119 | 1 | 0 | 0 |
| gtsrb | **146** (was 0) | 9 | 0 | 2 |

**27 groups (26 with `43,44` + 1 with `0,43,44`) have a genuine multi-seed mean±std**, all anchored by the live sfl2 seed-43/44 sweeps. GTSRB now shows 146 single-seed (seed0) groups instead of 0 — real progress, but note per §2a that none of the 9 GTSRB `43,44` groups currently have a matching seed0 (different `multiple_servers` setting), so GTSRB's seed-0 and seed-43/44 pools remain disjoint by config until that's reconciled. The remaining single-seed groups are otherwise unbroken placeholders pending the sweeps.

**The `0,43,44` group (cifar10 clean, avg/avg) is now well-behaved**: seed0=61.73/3.43/59.36 (asr/acc/ra), seed43=60.74/2.79/58.70, seed44=60.57/5.00/56.94 — accuracy std dropped from the previous 8.96pp (round-500 artifact) to **0.63pp**, consistent with genuine seed-to-seed variance rather than a measurement bug.

## Files produced/updated

- `extract_seed_results.py` — added `seed_source` field, the ambiguous-run split (project-keyed), the round-200-history metrics fix (`metrics_for_run()`), and the GTSRB seed0 promotion logic (`load_paper_reference()`, `matches_reference()`, `promote_gtsrb_seed0()`).
- `seed_runs_raw.csv` — all 511 fetched runs, now including `metrics_source` per run.
- `seed_runs_incomplete.csv` — 50 excluded runs (§1).
- `seed_runs_ambiguous_sfl2_seed0.csv` — full unfiltered 164 sfl2/seed0 runs (unchanged by promotion — still the complete transparency record, including the promoted ones and the losing duplicates).
- `seed_runs_duplicates.csv` — unchanged, 2 rows (1 true duplicate pair: `q71qw284`/`65n6qqgd`).
- `seed_results_aggregated.csv` — 414 groups, mean±std over available seeds, plus per-seed columns and the new `seed0_provenance` column (blank for cifar10/mnist legacy source; `exact_paper_match` / `reference_exists_no_match` / `recency_fallback_no_reference` for GTSRB).
- `paper_reference_values.csv` — read-only input (coordinator-provided), not modified.
