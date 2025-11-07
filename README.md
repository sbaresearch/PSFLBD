# SFL Backdoors
Implementation of backdoors in Split Federated Learning environments

## Folder structure

* `src/`: Main source code for the backdoor attacks.
* `src/src_base`: Source code for the setup with attacks: BadNets, DBA, Neurotoxin and defenses: Alignins, MKrum,MMetric , Foolsgold, FeDDLAD
* `src/src_more`: Source code for the setup with attacks: BadNets, DBA, Neurotoxin and option to combine them with Model Replacement technique and defenses:  Alignins, MKrum,MMetric , Foolsgold, FeDDLAD

## Getting Started

This project requires a specific enviroment to run correclty. It can be recreated using the provided `.yaml` file.

## Project Overview

The repository is divided into 2 main source directories:

### 1. `src_base` 
Contains the primary implemenration of Split Federated Learning.
* **sfl.py / agent_split.py**: Core logic for the split learning protocol.
* **aggregation.py**: Implements various defense mechanisms (Alignins, MKrum, etc.).
* **models/**: Includes ResNet and VGG architectures adapted for split training.
* **data/**: Local storage for the datasets.

### 2. `src_more`
An extended version of the repository used for advanced experimentation and automated runs.
* **sfl.py / agent_split.py**: Core logic for the split learning protocol.
* **agent.py** : Agent implemetation in federated learning.
* **aggregation.py**: Implements various defense mechanisms (Alignins, MKrum, etc.).
* **models/**: Includes ResNet and VGG architectures adapted for split training.
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

## Additional parameters for FL setup

In the FL setup 2 more attack are implemented: optimized version of `A3FL` attack and custom made attack called `alignment`.

`A3FL` attack builds shadow models by coppying current global model and then training those copies to unlearn the poisoned behavior. It optimizes the trigger itself, with the goal that current global model predicts the attacker's target class, and the shadow models also still predict target class. It wants to make a trigger that survives fututre benign training correction, not just a trigger that works on the model right now.

`alignment ` attack is custom made attack. It is a multi-objective optimization problem. It adds a stealth constraint to the a3fl persistence logic. The goal is to ensure the malicious update is mathematically indistriguishable from a benign one. 
- the attacker first caluclates a Benign Reference Gradient, which is the direction an honest client would move
- the objective function optimizes the trigger agaisnt three simultanoues goals: accuracy, persistence and stealth
    - the cosine proximity forces the malicious gradient to point in the same direction as benign reference gradient
    - the orthogonality penalty penalizes malicious gradient if its weight updates have different signs than bening reference gradient, preventing the update from being an outlier
- The trigger is updates so that wehn the attacker finally performs weight training, the resulting model updates sites perfectly within the cluster of honest updates, bypassing defenses like AlignIns.



Parameters specified for a3fl attack:

|Parameter|Default|Description|
| :--- | :--- | :--- |
|--trigger_outter_epochs|10|Controls how many iterations the attack spends "polishing" the trigger pixels.|
|--lambda_a3fl|0.5|Balances the current attack performance against the "persistence" across shadow models.|
|--trigger_lr|0.01|The step size for the sign-based gradient update of the trigger pixels.|

Parameters specified for alignement attack:
|Parameter|Default|Description|
| :--- | :--- | :--- |
|--aligment_alpha|0.1|Directional Stealth: Uses Cosine Similarity to force the poisoned gradient to point in the same direction as the benign reference.|
|--aligment_beta|0.01|Sign Stealth: Penalizes gradients that have a different mathematical sign than the benign model (prevents "conflicting" updates).|
|--trigger_outter_epochs|10|Controls how many iterations the attack spends "polishing" the trigger pixels.|
|--trigger_lr|0.01| The step size for the sign-based gradient update of the trigger pixels.|


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

