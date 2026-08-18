import utils
import models
import math

import copy
import numpy as np
from agent_split import Agent, FullSplitForEval
from aggregation import Aggregation
import torch
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
import logging
import argparse
import os
import warnings
from models import MainServer, SplitModel, get_split_models
import wandb
import pandas as pd
from utils import *
from tqdm import tqdm
warnings.filterwarnings("ignore")


class _NullWandbConfig:
    """Stand-in for wandb.config when wandb is disabled -- looks empty."""
    def items(self):
        return []

    def __contains__(self, key):
        return False

    def update(self, *args, **kwargs):
        pass


class _NullWandbRun:
    name = None


class _NullWandb:
    """Stand-in for the wandb module when no credentials/project were given
    and we're not running under a `wandb agent` sweep -- every call is a
    no-op so the rest of the script doesn't need `if use_wandb` checks."""
    def __init__(self):
        self.config = _NullWandbConfig()
        self.run = _NullWandbRun()

    def init(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    best_acc = -1
    best_asr = -1
    best_bcdr_acc = -1

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description="pass in a parameter")

    parser.add_argument(
        "--data", type=str, default="cifar10", help="dataset we want to train on"
    )
    parser.add_argument("--num_agents", type=int, default=20, help="number of agents:K")
    parser.add_argument(
        "--agent_frac", type=float, default=1.0, help="fraction of agents per round:C"
    )
    parser.add_argument(
        "--num_corrupt", type=int, default=2, help="number of corrupt agents"
    )
    parser.add_argument(
        "--model", type=str, default="resnet18", help="model architecture"
    )
    parser.add_argument(
        "--rounds", type=int, default=150, help="number of communication rounds:R"
    )
    parser.add_argument(
        "--local_ep", type=int, default=1, help="number of local epochs:E"
    )
    parser.add_argument("--bs", type=int, default=64, help="local batch size: B")
    parser.add_argument(
        "--client_lr", type=float, default=0.01, help="clients learning rate"
    )
    parser.add_argument(
        "--server_lr", type=float, default=0.01, help="servers learning rate"
    )
    parser.add_argument(
        "--target_class", type=int, default=7, help="target class for backdoor attack"
    )
    parser.add_argument(
        "--poison_frac",
        type=float,
        default=0.5,
        help="fraction of dataset to corrupt for backdoor attack",
    )
    parser.add_argument(
        "--pattern_type", type=str, default="plus", help="shape of bd pattern"
    )
    parser.add_argument(
        "--theta", type=int, default=8, help="break ties when votes sum to 0"
    )
    parser.add_argument(
        "--theta_ld", type=int, default=10, help="break ties when votes sum to 0"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed"
    )
    parser.add_argument(
        "--snap", type=int, default=1, help="do inference in every num of snap rounds"
    )
    parser.add_argument(
        "--device",
        default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="To use cuda, set to a specific GPU ID.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="num of workers for multithreading"
    )
    parser.add_argument(
        "--dense_ratio",
        type=float,
        default=0.25,
        help="num of workers for multithreading",
    )
    parser.add_argument(
        "--anneal_factor",
        type=float,
        default=0.0001,
        help="num of workers for multithreading",
    )
    parser.add_argument(
        "--se_threshold",
        type=float,
        default=1e-4,
        help="num of workers for multithreading",
    )
    parser.add_argument("--non_iid", action="store_true", default=False)
    parser.add_argument("--multiple_servers", type=bool, default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--attack",
        type=str,
        default="badnet",
        choices=["badnet", "DBA", "neurotoxin"],
    )
    parser.add_argument(
        "--aggr",
        type=str,
        default="avg",
        choices=[
            "avg",
            "alignins",
            "mkrum",
            "mmetric",
            "foolsgold",
            "feddlad"
        ],
        help="aggregation function to aggregate agents' local weights",
    )
    parser.add_argument(
        "--aggr_server",
        type=str,
        default="avg",
        choices=[
            "avg",
            "alignins",
            "mkrum",
            "mmetric",
            "foolsgold",
            "feddlad",
            "same"
        ],
    )
    parser.add_argument("--defense_start_round", type=int, default=1, help="Round at which defense mechanisms start")
    parser.add_argument("--lr_decay", type=float, default=0.99)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--mask_init", type=str, default="ERK")
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--same_mask", type=int, default=1)
    parser.add_argument("--cease_poison", type=float, default=100000)
    parser.add_argument("--exp_name_extra", type=str, help="defence name", default="")
    parser.add_argument("--super_power", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument("--lambda_s", type=float, default=1.0)
    parser.add_argument("--lambda_c", type=float, default=1.0)
    parser.add_argument("--split_point", type=str, default="layer1")
    parser.add_argument(
        "--attack_mode",
        type=str,
        default="baseline",
        choices=["baseline", "replacement"],
    )
    parser.add_argument("--bg", type=int, default = 3)
    parser.add_argument("--pg", type=int, default=0)
    parser.add_argument("--n_neighbors", type=int, default=0)
    parser.add_argument("--iqr_scale", type=float, default=0.5)
    parser.add_argument("--wandb_entity", type=str, default=None,
        help="wandb entity to log to. If neither this nor --wandb_project is set "
             "(and we're not running under `wandb agent`, i.e. a sweep), wandb "
             "logging is skipped entirely -- no account/credentials needed.")
    parser.add_argument("--wandb_project", type=str, default=None,
        help="wandb project to log to (see --wandb_entity).")

    args = parser.parse_args()

    if args.clean:
        args.num_corrupt = 0
        args.exp_name_extra = "clean"

    if args.super_power:
        args.exp_name_extra = "sp"

    # wandb is entirely optional: only turn it on if the user asked for it
    # (--wandb_entity/--wandb_project, e.g. via a sweep YAML's `entity`/
    # `project` keys) or we're already running as a `wandb agent` sweep
    # worker (which sets these env vars before invoking this script).
    use_wandb = bool(args.wandb_entity) or bool(args.wandb_project) or bool(
        os.environ.get("WANDB_SWEEP_ID") or os.environ.get("WANDB_RUN_ID")
    )
    if use_wandb:
        wandb.init(
            entity=args.wandb_entity or "adavarts-tuwien",
            project=args.wandb_project or "sfl2",
        )
    else:
        wandb = _NullWandb()

    per_data_dict = {
        "rounds": {"mnist": 200,"ets":200, "gtsrb": 200, "cifar10": 200, "cifar100": 200},
        "num_target": {"mnist": 10, "ets": 164, "gtsrb":43, "cifar10": 10, "cifar100": 100},
    }

    for key, value in wandb.config.items():
        setattr(args, key, value)

    if "rounds" not in wandb.config:
        args.rounds = per_data_dict["rounds"][args.data]
    args.num_target = per_data_dict["num_target"][args.data]
    if args.aggr_server == "same":
        args.aggr_server = args.aggr

    if args.model == "googlenet":
        args.client_lr = 0.001
        args.server_lr = 0.001

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.log_dir = utils.setup_logging(args)
    wandb.config.update(vars(args), allow_val_change=True)
    wandb.run.name = f"{args.data}_{args.model}_{args.attack}_{args.aggr_server}_{args.aggr}_{args.attack_mode}"

    # 2. Setup CSV Logger
    csv_file = os.path.join(args.log_dir, "results.csv")
    results_history = []

    train_dataset, val_dataset = utils.get_datasets(args.data)
    backdoor_train_dataset = None
    print(args.rounds)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    if args.non_iid:
        user_groups = utils.distribute_data_dirichlet(train_dataset, args)
    else:
        user_groups = utils.distribute_data(
            train_dataset, args, n_classes=args.num_target
        )

    idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()

    if args.data not in ("ets", "gtsrb"):
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    else:
        poisoned_val_set = utils.DatasetSplit(
            copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args
        )

    poisoned_val_loader = DataLoader(
        poisoned_val_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    if args.data not in ("ets", "gtsrb"):
        idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
        poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(
            poisoned_val_set_only_x.dataset,
            args,
            idxs,
            poison_all=True,
            modify_label=False,
        )
    else:
        poisoned_val_set_only_x = utils.DatasetSplit(
            copy.deepcopy(val_dataset),
            idxs,
            runtime_poison=True,
            args=args,
            modify_label=False,
        )

    poisoned_val_only_x_loader = DataLoader(
        poisoned_val_set_only_x,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # initialize a model, and the agents
    client_part, server_part, _ = models.get_split_models(args.data, args)
    global_client_model = client_part.to(args.device)

    main_server = MainServer(server_part, args.num_agents, args, device=args.device,
                             label_smoothing=0.1 if args.data == "femnist" else 0.0,
                             multiple_servers=args.multiple_servers)


    global_mask = {}
    neurotoxin_mask = {}
    updates_dict = {} #stores clients' updates per round
    n_model_params = len(
        parameters_to_vector(
            [global_client_model.state_dict()[name] for name in global_client_model.state_dict()]
        )
    )
    params = {
    name: copy.deepcopy(global_client_model.state_dict()[name])
    for name in global_client_model.state_dict()
    }

    malicious_client_ids = list(range(args.num_corrupt)) # find which clients are corrupted
   
    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):#builds all clients and assigns their data shards
        agent = Agent( #this is a client in federated learning
            _id,
            args,
            train_dataset,
            user_groups[_id],
            backdoor_train_dataset=backdoor_train_dataset,
            main_server = main_server
        )
        agent.is_malicious = 1 if _id < args.num_corrupt else 0
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

        logging.info(
            "build client:{} mal:{} data_num:{}".format(
                _id, agent.is_malicious, agent.n_data
            )
        )

    aggregator = Aggregation(agent_data_sizes, n_model_params, args) #creates server side aggregation

    criterion = nn.CrossEntropyLoss().to(args.device)
    agent_updates_dict = {}

    best_acc = -1

    for rnd in tqdm(range(1, args.rounds + 1)):
        logging.info("--------round {} ------------".format(rnd))

        rnd_global_client_params = parameters_to_vector(
            [copy.deepcopy(global_client_model.state_dict()[name]) for name in global_client_model.state_dict()]
        ).detach()

        agent_updates_dict = {}
        agent_server_updates = {}

        chosen = np.random.choice(
            args.num_agents,
            math.floor(args.num_agents * args.agent_frac),
            replace=False,
        )
        chosen = sorted(chosen)

        for agent_id in chosen:
            if agents[agent_id].is_malicious and args.super_power:
                continue

            # SplitFedV1: returns both client and server deltas
            client_update, server_update = agents[agent_id].local_train_splitfedv1(
                global_client_model,
                main_server,
                round=rnd,
                neurotoxin_mask=neurotoxin_mask,
            )
            agent_updates_dict[agent_id] = client_update
            agent_server_updates[agent_id] = server_update

            # IMPORTANT: reset client model back to round start for next agent
            utils.vector_to_model(copy.deepcopy(rnd_global_client_params), global_client_model)

        mal_norms = []
        hon_norms = []
        for a_id in chosen: # Use a different iterator name to be safe
            norm = torch.norm(agent_updates_dict[a_id]).item()
            if agents[a_id].is_malicious:
                mal_norms.append(norm)
            else:
                hon_norms.append(norm)

        if len(mal_norms) > 0 and len(hon_norms) > 0:
            avg_mal = sum(mal_norms) / len(mal_norms)
            avg_hon = sum(hon_norms) / len(hon_norms)
            print(f"\n[GRADIENT DEBUG Rnd {rnd}]")
            print(f"Avg Malicious Delta Norm: {avg_mal:.6f}")
            print(f"Avg Honest Delta Norm:    {avg_hon:.6f}")
            print(f"Power Ratio (Mal/Hon):    {avg_mal/avg_hon:.4f}")
            if avg_hon > avg_mal * 2:
                print("!!! ALERT: Malicious signal is too quiet. FedAvg will wash this out.")

        # 2) Aggregate/defend server-side updates (only for SplitFed path)
        main_server.aggregate_server_updates_round(
            client_ids=chosen,
            agent_server_updates=agent_server_updates,
            agent_data_sizes=agent_data_sizes,
            rnd=rnd,
        )

        # 1. Get the result from the aggregator
        aggregated_result, neurotoxin_mask = aggregator.aggregate_updates(
            global_client_model,
            agent_updates_dict,
            rnd=rnd
        )

        # 2. Check the type. If it's already a dict, you don't need vector_to_name_param!
        if isinstance(aggregated_result, dict):
            updates_dict = aggregated_result
        else:
            # If it's a flat vector, then convert it
            updates_dict = vector_to_name_param(
                aggregated_result, 
                copy.deepcopy(global_client_model.state_dict())
            )

        # Optional debug
        sd0 = main_server.server_models[0].state_dict()
        # sd1 = main_server.server_models[1].state_dict()
        k = next(iter(sd0.keys()))
        # print("server branch diff", torch.norm(sd0[k] - sd1[k]).item())


        # inference in every args.snap rounds
        logging.info("---------Test {} ------------".format(rnd))
        if rnd % args.snap == 0:
            eval_model = FullSplitForEval(
                copy.deepcopy(global_client_model).to(args.device),
                copy.deepcopy(main_server.server_models[0]).to(args.device),
            ).to(args.device)
            eval_model.eval()
            with torch.no_grad():
                x_dbg, y_dbg = next(iter(val_loader))
                x_dbg = x_dbg.to(args.device)
                y_dbg = y_dbg.to(args.device)

                logits = eval_model(x_dbg)  # shape [B, num_classes]
                probs = torch.softmax(logits, dim=1)

                print(f"[DBG rnd={rnd}] logits mean={logits.mean().item():.4f}, std={logits.std().item():.4f}, "
                    f"maxlogit mean={logits.max(dim=1).values.mean().item():.4f}")
                print(f"[DBG rnd={rnd}] probs max mean={probs.max(dim=1).values.mean().item():.4f}, "
                    f"pred entropy mean={(- (probs * (probs+1e-12).log()).sum(dim=1)).mean().item():.4f}")
                print(f"[DBG rnd={rnd}] preds bincount={torch.bincount(logits.argmax(dim=1), minlength=args.num_target).cpu().tolist()}")
            # -------------------------------------
            val_acc = utils.get_loss_n_accuracy(eval_model, criterion, val_loader, args, rnd, args.num_target)
            asr = utils.get_loss_n_accuracy(eval_model, criterion, poisoned_val_loader, args, rnd, num_classes=args.num_target)
            poison_acc = utils.get_loss_n_accuracy(eval_model, criterion, poisoned_val_only_x_loader, args, rnd, args.num_target)

            del eval_model

            logging.info("Clean ACC:              %.4f" % val_acc)
            logging.info("Attack Success Ratio:   %.4f" % asr)
            logging.info("Robustness ACC:           %.4f" % poison_acc)

            print("Clean ACC:              %.4f" % val_acc)
            print("Attack Success Ratio:   %.4f" % asr)
            print("Robustness ACC:           %.4f" % poison_acc)

            wandb.log({
                "round": rnd,
                "clean_acc": val_acc,
                "asr": asr,
                "robustness acc": poison_acc,
            })

            # 5. Log to CSV
            results_history.append({
                "round": rnd, 
                "clean_acc": val_acc, 
                "asr": asr,
                "robustness acc": poison_acc,
            })
            pd.DataFrame(results_history).to_csv(csv_file, index=False)
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_asr = asr
                best_bcdr_acc = poison_acc
            

        logging.info("------------------------------".format(rnd))

    logging.info("Best results:")
    logging.info("Clean ACC:              %.4f" % best_acc)
    logging.info("Attack Success Ratio:   %.4f" % best_asr)
    logging.info("Robustness ACC:           %.4f" % best_bcdr_acc)
    logging.info("Training has finished!")
    wandb.finish()
