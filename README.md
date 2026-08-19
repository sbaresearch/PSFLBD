# Can Federated Learning Backdoor Defenses Protect Parallel Split-Federated Learning?

This repository contains the code accompanying the paper **"Can Federated Learning Backdoor
Defenses Protect Parallel Split-Federated Learning?"**

With the growing use of collaborative learning paradigms in distributed edge applications, hybrid approaches such as Split-Federated Learning (SFL) have emerged to combine the benefits of federated and split learning. 
However, despite the increasing adoption of SFL to improve the trustworthiness of collaborative AI systems, the security of these systems has not been systematically studied, particularly with respect to backdoor attacks known to affect federated learning (FL). 
While prior work has demonstrated that SFL is vulnerable to such attacks, it remains unexplored whether state-of-the-art FL defenses provide reliable protection in this setting.

In this work, we define multiple realistic threat models based on trust assumptions and the timing of defense deployment. We then analyze representative FL backdoor defenses in the parallel SFL setting under these models.
Our findings show that in SFL, defense effectiveness is highly dependent on the entity responsible for defense deployment. 
Among the considered methods, only one, namely \textit{AlignIns}, is found effective across most threat models. 
Furthermore, in delayed-defense settings where the backdoor is already (at least partially) embedded into the model, defense effectiveness strongly depends on dataset complexity.

## Contents:

- **Backdoor attacks**: BadNets, DBA, and Neurotoxin, optionally combined with the Model
  Replacement technique (`--attack_mode replacement`).
- **Defenses**: AlignIns, MKrum, MMetric, FoolsGold, and FedDLAD, applied independently on the
  Client Aggregation Server and Main Server.
- **Datasets/models**: MNIST and CIFAR-10 (ResNet-18), and GTSRB (GoogLeNet).
- A **sweeps/** folder of wandb sweep configs covering the full attack x defense x seed grid
  used in the paper, and a **results/** pipeline that pulls run data from wandb, aggregates it
  across seeds, and renders it into the paper's LaTeX tables and figures.

## Folder structure

* `src/`: Environment setup (`setup_env.sh`, `requirements.txt`) and main source code.
* `src/src_more/`: Source code for the SFL/FL implementation, attacks, and defenses.
  * `sfl.py`: Entry point for Split Federated Learning experiments.
  * `federated.py`: Entry point for the standard FL baseline.
  * `agent_split.py` / `agent.py`: Client-side training logic for SFL and FL, respectively.
  * `models.py`: Server-side split-model logic and update aggregation across rounds.
  * `aggregation.py`: Defense implementations (AlignIns, MKrum, MMetric, FoolsGold, FedDLAD).
  * `resnet.py`: ResNet/GoogLeNet architectures adapted for split training.
  * `utils.py`: Data loading, poisoning, and logging utilities.
  * `sweeps/`: wandb sweep configs (`base/` for the main attack x defense grid, `round/` for
    the defense-start-round experiments).
  * `results/`: Scripts and notebook used to extract, aggregate, and render the paper's tables
    and figures from wandb.

## Getting Started

```
cd src
./setup_env.sh          # creates a conda env named "sfl-backdoors" with everything installed
conda activate sfl-backdoors
```

This installs torch/torchvision from PyTorch's cu121 wheel index (see `requirements.txt`) rather than via conda, and is the only verified setup for GTSRB: torchvision must be >=0.15 for its built-in `datasets.GTSRB` loader.

## Configuration and Parameters

|Parameter|Default|Description|
| :--- | :--- | :--- |
|--data|cifar10|Dataset to train on (cifar10, mnist, gtsrb).|
|--model|resnet18|Model architecture (resnet18, googlenet).|
|--num_agents|20|Total number of participating agents (K).|
|--num_corrupt|2|Number of malicious/adversarial agents.|
|--poison_frac|0.5|Fraction of a corrupt agent's data that is poisoned.|
|--attack|badnet|Attack type: badnet, DBA, neurotoxin.|
|--attack_mode|baseline|baseline or replacement (Model Replacement); SFL only.|
|--aggr|avg|Client-side aggregation/defense: avg, alignins, mkrum, foolsgold, mmetric, feddlad.|
|--aggr_server|avg|Server-side aggregation/defense (SFL only); same choices as `--aggr`, plus `same` to mirror `--aggr`.|
|--defense_start_round|1|Round at which the chosen defense(s) start being applied.|
|--rounds|150|Total communication rounds (R).|
|--target_class|7|The target class for the backdoor attack.|
|--split_point|layer1|The layer where the model is split between client and server (SFL only).|
|--multiple_servers|False|Split server-side computation across multiple processes (SFL only; memory optimization, no effect on results).|
|--seed|0|Random seed.|
|--clean|False|Run without any corrupt agents (clean baseline).|
|--wandb_entity / --wandb_project|None|Optional wandb logging. If neither is set (and this isn't running under `wandb agent`), wandb is skipped entirely -- no account needed.|

## How to Run

### SFL

**Basic experiment** -- 20 agents, AlignIns defense on the MS (server side), CIFAR-10:
```
python sfl.py --num_agents 20 --aggr_server alignins --data cifar10
```

**Backdoor attack only** -- BadNets with 4 corrupt agents targeting class 7:
```
python sfl.py --attack badnet --num_corrupt 4 --target_class 7 --poison_frac 0.5
```

**Attack + Model Replacement + defense** -- DBA with 4 corrupt agents, MKrum on both server, GTSRB:
```
python sfl.py --data gtsrb --model googlenet --attack DBA --attack_mode replacement --num_corrupt 4 --num_agents 20 --aggr mkrum --aggr_server mkrum --target_class 7 --poison_frac 0.5
```

### FL

**Basic experiment**:
```
python federated.py --num_agents 20 --aggr_server alignins --data cifar10
```

**Backdoor attack only**:
```
python federated.py --attack badnet --num_corrupt 5 --target_class 7 --poison_frac 0.5
```

**Attack + defense**:
```
python federated.py --attack badnet --num_corrupt 4 --num_agents 20 --aggr_server mkrum --target_class 7 --poison_frac 0.5
```

### Reproducing the paper's experiment grid (wandb sweeps)

`sweeps/base/` contains one sweep config per dataset covering the full attack x attack_mode x defense x seed grid (plus a `clean_*.yaml` companion for the no-attack baseline);
`sweeps/round/` contains the defense-start-round experiments. To launch one:
```
cd src/src_more
wandb sweep sweeps/base/all_MN.yaml --entity <your-entity> --project <your-project>
wandb agent <sweep-id-printed-above>
```

### Reproducing the paper's tables and figures

```
cd src/src_more/results
python extract_seed_results.py --project <your-entity>   # pulls runs from wandb, aggregates across seeds
python generate_latex_tables.py                           # renders table{1,2,3,4}_*.tex from the aggregated CSV
```

`plotting_seed_avg.ipynb` regenerates the defense-start-round figures the same way; fill in your
own wandb project name(s) in its first cell (`PROJECTS = [...]`), then run it top to bottom.

## Citation

If you use this code, please cite:

> Can Federated Learning Backdoor Defenses Protect Parallel Split-Federated Learning?
