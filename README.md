# SFL Backdoors
Implementation of backdoors in Split Federated Learning environments

## Folder structure

* `src/`: Main source code for the backdoor attacks.
* `src/src_more`: Source code for the setup with attacks: BadNets, DBA, Neurotoxin and option to combine them with Model Replacement technique and defenses:  Alignins, MKrum, MMetric, Foolsgold, FeDDLAD.

## Getting Started

```
cd src
./setup_env.sh          # creates a conda env named "sfl-backdoors" with everything installed
conda activate sfl-backdoors
```

This installs torch/torchvision from PyTorch's cu121 wheel index (see `requirements.txt`) rather than
via conda, and is the only verified setup for GTSRB: torchvision must be >=0.15 for its built-in
`datasets.GTSRB` loader, otherwise `src_more/utils.py` falls back to a custom loader with a known
filename bug and GTSRB training will crash. (An older `env.yaml`, pinned to torch 1.8 / torchvision
0.9, predated this fix and has been removed.)

## Project Overview

### `src_more`
Contains the primary implemenration of Split Federated Learning with an extended version of the repository used for advanced experimentation and automated runs.
* **sfl.py / agent_split.py**: Core logic for the split learning protocol.
* **agent.py** : Agent implemetation in federated learning.
* **aggregation.py**: Implements various defense mechanisms (Alignins, MKrum, etc.).
* **models/**: Includes ResNet and GoogLeNet architectures adapted for split training.
* **data/**: Local storage for the datasets.
* **federated.py**: Standard FL orchestration to compare against the Split (SFL) approach.


## Configuration and Parameters

|Parameter|Default|Description|
| :--- | :--- | :--- |
|--data|cifar10|Dataset to train on (e.g., cifar10, mnist).|
|--num_agents|20|Total number of participating agents (K).|
|--num_corrupt|2|Number of malicious/adversarial agents.|
|--poison_frac|0.5|Ratio of data posioning on client. |
|--attack|badnet|Attack type: badnet, DBA, neurotoxin.|
|--attack_mode|baseline|Attack type: baseline,replacement, exists only in SFL setup.|
|--aggr|avg|Aggregation mechanism on Federated Server in SFL or on Server in FL: alignins, mkrum, foolsgold, feddlad, etc.|
|--aggr_server|avg|Aggregation mechanism on Federated Server in SFL or on Server in FL: alignins, mkrum, foolsgold, feddlad, etc.|
|--rounds|150|Total communication rounds (R).|
|--target_class|7|The target class for the backdoor attack.|
|--split_point|layer1|The layer where the model is split between client and server.|

## How to Run

### SFL
#### 1. Running a Basic Experiment

To run a standard Split Federated Learning session with 20 agents and the alignins defense:
python sfl.py --num_agents 20 --aggr_server alignins --data cifar10

#### 2. Running a Backdoor Attack

To simulate a BadNet attack with 5 corrupt agents targeting class 7:
python sfl.py --attack badnet --num_corrupt 5 --target_class 7 --poison_frac 0.5

#### 2. Running a Combined Attack and Defense

To simulate a DBA attack with 4 corrupt agents, 20 agents, targeting class 7, and combine that with model replacement, and MKrum defense:
python sfl.py --attack badnet --attack_mode replacement --num_corrupt 4 --num_agents 20 --defense mkrum --target_class 7 --poison_frac 0.5

### FL

#### 1. Running a Basic Experiment

To run a standard Federated Learning session with 20 agents and the alignins defense:
python federated.py --num_agents 20 --aggr_server alignins --data cifar10

#### 2. Running a Backdoor Attack

To simulate a BadNet attack with 5 corrupt agents targeting class 7:
python federated.py --attack badnet --num_corrupt 5 --target_class 7 --poison_frac 0.5

#### 2. Running a  Attack and Defense

To simulate a DBA attack with 4 corrupt agents, 20 agents, targeting class 7, and MKrum defense:
python sfl.py --attack badnet  --num_corrupt 4 --num_agents 20 --defense mkrum --target_class 7 --poison_frac 0.5

