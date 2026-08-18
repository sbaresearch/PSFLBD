import utils
import models
import math
import copy
import numpy as np
from agent import Agent
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
import wandb

warnings.filterwarnings("ignore")


class _NullWandb:
    """Stand-in for the wandb module when no credentials/project were given
    and we're not running under a `wandb agent` sweep -- every call is a
    no-op so the rest of the script doesn't need `if use_wandb` checks."""
    def init(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


if __name__ == "__main__":
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
        "--rounds", type=int, default=150, help="number of communication rounds:R"
    )
    parser.add_argument(
        "--local_ep", type=int, default=2, help="number of local epochs:E"
    )
    parser.add_argument("--bs", type=int, default=64, help="local batch size: B")
    parser.add_argument(
        "--client_lr", type=float, default=0.1, help="clients learning rate"
    )
    parser.add_argument(
        "--server_lr", type=float, default=1, help="servers learning rate"
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
    parser.add_argument("--debug", action="store_true", default=False)
    # parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--attack",
        type=str,
        default="badnet",
        choices=["badnet", "DBA", "neurotoxin"],
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
        ],
        help="aggregation function to aggregate agents' local weights",
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
        ],
        help="aggregation function to aggregate agents' local weights",
    )
    parser.add_argument("--lr_decay", type=float, default=0.99)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--mask_init", type=str, default="ERK")
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--same_mask", type=int, default=1)
    parser.add_argument("--cease_poison", type=float, default=100000)
    # parser.add_argument("--exp_name_extra", type=str, help="defence name", default="")
    parser.add_argument("--super_power", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument("--lambda_s", type=float, default=1.0)
    parser.add_argument("--lambda_c", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--wandb_entity", type=str, default=None,
        help="wandb entity to log to. If neither this nor --wandb_project is set "
             "(and we're not running under `wandb agent`, i.e. a sweep), wandb "
             "logging is skipped entirely -- no account/credentials needed.")
    parser.add_argument("--wandb_project", type=str, default=None,
        help="wandb project to log to (see --wandb_entity).")


    args = parser.parse_args()

    use_wandb = bool(args.wandb_entity) or bool(args.wandb_project) or bool(
        os.environ.get("WANDB_SWEEP_ID") or os.environ.get("WANDB_RUN_ID")
    )
    if use_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project or ("FL-2" if args.debug else "FL-2"),
            config=vars(args),
            name=f"DEBUG_{args.exp_name_extra}" if args.debug else args.exp_name_extra,
        )
    else:
        wandb = _NullWandb()


    
    if args.clean:
        args.num_corrupt = 0
        args.exp_name_extra = "clean"

    if args.super_power:
        args.exp_name_extra = "sp"

    per_data_dict = {
        "rounds": {"mnist": 100, "cifar10": 300, "cifar100": 300},
        "num_target": {"mnist": 10, "cifar10": 10, "cifar100": 100},
    }

    args.rounds = per_data_dict["rounds"][args.data]
    args.num_target = per_data_dict["num_target"][args.data]

    args.log_dir = utils.setup_logging(args)

    train_dataset, val_dataset = utils.get_datasets(args.data)
    backdoor_train_dataset = None

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

    if args.data != "gtsrb":
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
    if args.data != "gtsrb":
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
    global_model = models.get_model(args.data, args).to(args.device)

    neurotoxin_mask = {}
    updates_dict = {}
    n_model_params = len(
        parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]
        )
    )
    params = {
        name: copy.deepcopy(global_model.state_dict()[name])
        for name in global_model.state_dict()
    }

    malicious_client_ids = list(range(args.num_corrupt))
    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):
        agent = Agent(
            _id,
            args,
            train_dataset,
            user_groups[_id],
            backdoor_train_dataset=backdoor_train_dataset,
        )
        agent.is_malicious = 1 if _id < args.num_corrupt else 0
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

        logging.info(
            "build client:{} mal:{} data_num:{}".format(
                _id, agent.is_malicious, agent.n_data
            )
        )

    aggregator = Aggregation(agent_data_sizes, n_model_params, args)

    criterion = nn.CrossEntropyLoss().to(args.device)
    agent_updates_dict = {}

    best_acc = -1

    for rnd in range(1, args.rounds + 1):
        logging.info("--------round {} ------------".format(rnd))
        rnd_global_params = parameters_to_vector(
            [
                copy.deepcopy(global_model.state_dict()[name])
                for name in global_model.state_dict()
            ]
        )
        agent_updates_dict = {}
        chosen = np.random.choice(
            args.num_agents,
            math.floor(args.num_agents * args.agent_frac),
            replace=False,
        )
        chosen = sorted(chosen)

        for agent_id in chosen:
            if agents[agent_id].is_malicious and args.super_power:
                continue
            global_model = global_model.to(args.device)

            update = agents[agent_id].local_train(
                global_model, criterion, rnd, neurotoxin_mask=neurotoxin_mask
            )
            agent_updates_dict[agent_id] = update
            utils.vector_to_model(copy.deepcopy(rnd_global_params), global_model)

        # aggregate params obtained by agents and update the global params
        updates_dict, neurotoxin_mask = aggregator.aggregate_updates(
            global_model, agent_updates_dict
        )

        # inference in every args.snap rounds
        logging.info("---------Test {} ------------".format(rnd))
        if rnd % args.snap == 0:
            val_acc = utils.get_loss_n_accuracy(
                global_model, criterion, val_loader, args, rnd, args.num_target
            )
            asr = utils.get_loss_n_accuracy(
                global_model,
                criterion,
                poisoned_val_loader,
                args,
                rnd,
                num_classes=args.num_target,
            )
            poison_acc = utils.get_loss_n_accuracy(
                global_model,
                criterion,
                poisoned_val_only_x_loader,
                args,
                rnd,
                args.num_target,
            )

            logging.info("Clean ACC:              %.4f" % val_acc)
            logging.info("Attack Success Ratio:   %.4f" % asr)
            logging.info("Robustness ACC:           %.4f" % poison_acc)
            wandb.log({
                "round": rnd,
                "eval/clean_acc": val_acc,
                "eval/attack_success_ratio": asr,
                "eval/robustness_acc": poison_acc,
                "eval/best_clean_acc": best_acc
            })
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
