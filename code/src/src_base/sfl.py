import utils
import models
import math

import copy
import numpy as np
from agent_split import Agent, FullSplitForEval
# from agent_sparse import Agent as Agent_s
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
warnings.filterwarnings("ignore")



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
        "--rounds", type=int, default=150, help="number of communication rounds:R"
    )
    parser.add_argument(
        "--local_ep", type=int, default=1, help="number of local epochs:E"
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
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--attack",
        type=str,
        default="badnet",
        choices=["badnet", "DBA", "neurotoxin", "pgd"],
    )
    parser.add_argument(
        "--aggr",
        type=str,
        default="avg",
        choices=[
            "avg",
            "alignins",
            "rlr",
            "mkrum",
            "mmetric",
            "lockdown",
            "foolsgold",
            "rfa",
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
            "rlr",
            "mkrum",
            "mmetric",
            "lockdown",
            "foolsgold",
            "rfa",
            "feddlad"
        ],
        help="aggregation function to aggregate agents' local weights",
    )
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

    args = parser.parse_args()

    if args.clean:
        args.num_corrupt = 0
        args.exp_name_extra = "clean"

    if args.super_power:
        args.exp_name_extra = "sp"

    per_data_dict = {
        "rounds": {"mnist": 25,"fmnist": 50, "cifar10": 100, "cifar100": 100, "tinyimagenet": 50,"sen140": 50},
        "num_target": {"mnist": 10,"fmnist": 10, "cifar10": 10, "cifar100": 100, "tinyimagenet": 200,"sen140": 2},
    }

    args.rounds = per_data_dict["rounds"][args.data]
    args.num_target = per_data_dict["num_target"][args.data]

    args.log_dir = utils.setup_logging(args)
    wandb.init(
        project="split-fed-backdoor-2", 
        config=args, 
        name=f"{args.data}_{args.attack}_{args.aggr_server}_{args.aggr}"
    )
    
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

    if args.data != "tinyimagenet":
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
    if args.data != "tinyimagenet":
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
    # global_model = models.get_model(args.data, args).to(args.device) #this is the server model, central model
    client_part, server_part, _ = models.get_split_models(args.data, args)
    global_client_model = client_part.to(args.device)

    main_server = MainServer(server_part, args.num_agents, args, device=args.device)


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

    if args.aggr == "lockdown":
        sparsity = utils.calculate_sparsities(args, params, distribution=args.mask_init)
        mask = utils.init_masks(params, sparsity)
    malicious_client_ids = list(range(args.num_corrupt)) # find which clients are corrupted
   
    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):#builds all clients and assigns their data shards
        if args.aggr == "lockdown":
            if args.same_mask == 0:
                agent = Agent_s(
                    _id,
                    args,
                    train_dataset,
                    user_groups[_id],
                    mask=utils.init_masks(params, sparsity),
                    backdoor_train_dataset=backdoor_train_dataset,
                )
            else:
                agent = Agent_s(
                    _id,
                    args,
                    train_dataset,
                    user_groups[_id],
                    mask=mask,
                    backdoor_train_dataset=backdoor_train_dataset,
                )
        else:
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

    # for rnd in range(1, args.rounds + 1):
    #     logging.info("--------round {} ------------".format(rnd))
    #     rnd_global_client_params = parameters_to_vector(
    #         [copy.deepcopy(global_client_model.state_dict()[name]) for name in global_client_model.state_dict()]
    #     )
    #     agent_updates_dict = {}
    #     agent_server_updates = {} 
    #     chosen = np.random.choice(
    #         args.num_agents,
    #         math.floor(args.num_agents * args.agent_frac),
    #         replace=False,
    #     )
    #     chosen = sorted(chosen)
    #     if args.aggr == "lockdown":
    #         old_mask = [copy.deepcopy(agent.mask) for agent in agents]

    #     for agent_id in chosen:
    #         if agents[agent_id].is_malicious and args.super_power:
    #             continue
    #         # global_model = global_model.to(args.device)

    #         if args.aggr == "lockdown":
    #             update = agents[agent_id].local_train(
    #                 global_model,
    #                 criterion,
    #                 rnd,
    #                 global_mask=global_mask,
    #                 neurotoxin_mask=neurotoxin_mask,
    #                 updates_dict=updates_dict,
    #             )
    #         else:
    #             # update = agents[agent_id].local_train_splitfedv1(
    #             #     global_client_model,
    #             #     main_server,
    #             #     round=rnd,
    #             #     neurotoxin_mask=neurotoxin_mask,
    #             # )
    #             client_update, server_update = agents[agent_id].local_train_splitfedv1(
    #                 global_client_model,
    #                 main_server,
    #                 round=rnd,
    #                 neurotoxin_mask=neurotoxin_mask,
    #             )
    #         # agent_updates_dict[agent_id] = update
    #         # utils.vector_to_model(copy.deepcopy(rnd_global_client_params), global_client_model)
    #             agent_updates_dict[agent_id] = client_update
    #             agent_server_updates[agent_id] = server_update
    #         utils.vector_to_model(copy.deepcopy(rnd_global_client_params), global_client_model)

    #     # aggregate params obtained by agents and update the global params
    #     updates_dict, neurotoxin_mask = aggregator.aggregate_updates(global_client_model, agent_updates_dict)
    #     after = parameters_to_vector([global_client_model.state_dict()[n] for n in global_client_model.state_dict()]).detach()
    #     # print("||global_client_change||", torch.norm(after - rnd_global_client_params).item())

    #     # main_server.fedavg_server_branches(chosen)
    #     main_server.aggregate_server_updates_round(
    #         client_ids=chosen,
    #         agent_server_updates=agent_server_updates,   # dict: {client_id: server_delta_dict}
    #         agent_data_sizes=agent_data_sizes
    #     )


    #     sd0 = main_server.server_models[0].state_dict()
    #     sd1 = main_server.server_models[1].state_dict()
    #     k = next(iter(sd0.keys()))
    #     # print("server branch diff", torch.norm(sd0[k] - sd1[k]).item())
    for rnd in range(1, args.rounds + 1):
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

        if args.aggr == "lockdown":
            old_mask = [copy.deepcopy(agent.mask) for agent in agents]

        for agent_id in chosen:
            if agents[agent_id].is_malicious and args.super_power:
                continue

            if args.aggr == "lockdown":
                # your existing lockdown client training
                client_update = agents[agent_id].local_train(
                    global_model,
                    criterion,
                    rnd,
                    global_mask=global_mask,
                    neurotoxin_mask=neurotoxin_mask,
                    updates_dict=updates_dict,
                )
                agent_updates_dict[agent_id] = client_update

            else:
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

        # 1) Aggregate/defend client-side updates (as before)
        updates_dict, neurotoxin_mask = aggregator.aggregate_updates(
            global_client_model,
            agent_updates_dict
        )

        # 2) Aggregate/defend server-side updates (only for SplitFed path)
        if args.aggr != "lockdown":
            main_server.aggregate_server_updates_round(
                client_ids=chosen,
                agent_server_updates=agent_server_updates,
                agent_data_sizes=agent_data_sizes,
            )

        # Optional debug
        sd0 = main_server.server_models[0].state_dict()
        sd1 = main_server.server_models[1].state_dict()
        k = next(iter(sd0.keys()))
        # print("server branch diff", torch.norm(sd0[k] - sd1[k]).item())


        # inference in every args.snap rounds
        logging.info("---------Test {} ------------".format(rnd))
        if rnd % args.snap == 0:
            if args.aggr != "lockdown":
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
            else:
                test_model = copy.deepcopy(global_model)

                # CF
                for name, param in test_model.named_parameters():
                    mask = 0
                    for id, agent in enumerate(agents):
                        mask += old_mask[id][name].to(args.device)
                    param.data = torch.where(
                        mask.to(args.device) >= args.theta_ld,
                        param,
                        torch.zeros_like(param),
                    )
                val_acc = utils.get_loss_n_accuracy(
                    test_model, criterion, val_loader, args, rnd, args.num_target
                )
                asr = utils.get_loss_n_accuracy(
                    test_model,
                    criterion,
                    poisoned_val_loader,
                    args,
                    rnd,
                    args.num_target,
                )
                poison_acc = utils.get_loss_n_accuracy(
                    test_model,
                    criterion,
                    poisoned_val_only_x_loader,
                    args,
                    rnd,
                    args.num_target,
                )
                del test_model

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