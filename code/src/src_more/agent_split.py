import copy
import time

import torch
import torch.nn as nn
import utils
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader

def freeze_bn(m):
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()

class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None, mask=None, backdoor_train_dataset=None, main_server = None):
        self.id = id
        self.args = args
        self.error = 0
        self.hessian_metrix = []
        self.main_server = main_server
        # get datasets, fedemnist is handled differently as it doesn't come with pytorch

        if self.args.data not in ("tinyimagenet", "gtsrb", "ets"):
            self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)

            # for backdoor attack, agent poisons his local dataset
            if self.id < args.num_corrupt and self.args.attack != 'non' and self.args.data != 'sen140':

                self.clean_backup_dataset = copy.deepcopy(train_dataset)
                self.data_idxs = data_idxs
                utils.poison_dataset(train_dataset, args, data_idxs, agent_idx=self.id)
            elif self.id < args.num_corrupt and self.args.attack != 'non' and self.args.data == 'sen140':
                self.clean_backup_dataset = copy.deepcopy(train_dataset)
                self.data_idxs = data_idxs
                benign_part = data_idxs[:int(len(data_idxs) * (1 - self.args.poison_frac))]
                malicious_part = data_idxs[int(len(data_idxs) * (1 - self.args.poison_frac)):]

                self.train_dataset = utils.DatasetSplit_new(train_dataset, backdoor_train_dataset, benign_part, malicious_part, data_idxs)
        else:
            self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs, runtime_poison=True, args=args,
                                                        client_id=id)
        # get dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                       num_workers=args.num_workers, pin_memory=False, drop_last=True)
        # size of local dataset
        self.n_data = len(self.train_dataset)
        self.is_malicious = self.id < args.num_corrupt
        self.clean_ref_loader = None
        self.shadow_server_model = None
        self.shadow_server_opt = None

        if self.is_malicious:
            if self.args.data not in ("tinyimagenet", "gtsrb", "ets"):
                clean_ref_dataset = utils.DatasetSplit(self.clean_backup_dataset, self.data_idxs)
                self.clean_ref_loader = DataLoader(
                    clean_ref_dataset,
                    batch_size=self.args.bs,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=False,
                    drop_last=True,
                )
            if args.data in ("mnist", "fmnist", "femnist"):
                img_shape = (1, 28, 28)
            elif args.data == "tinyimagenet":
                img_shape = (3, 64, 64)
            else:
                img_shape = (3, 32, 32)
            self.trigger = torch.ones((1, *img_shape), requires_grad=True, device=self.args.device) * 0.5
            self.mask = torch.zeros_like(self.trigger)
            s = self.args.trigger_size
            self.mask[:, :, 2:2+s, 2:2+s] = 1.0

    def _raise_if_nonfinite(self, name, value, round_idx, local_epoch, batch_idx):
        if torch.is_tensor(value):
            if torch.isfinite(value).all():
                return
            detached = value.detach()
            finite_mask = torch.isfinite(detached)
            finite_vals = detached[finite_mask]
            min_val = finite_vals.min().item() if finite_vals.numel() else float("nan")
            max_val = finite_vals.max().item() if finite_vals.numel() else float("nan")
            raise RuntimeError(
                f"Non-finite tensor detected in {name} "
                f"(client={self.id}, round={round_idx}, epoch={local_epoch}, batch={batch_idx}, "
                f"shape={tuple(detached.shape)}, min_finite={min_val}, max_finite={max_val})"
            )
        if isinstance(value, (float, int)) and not torch.isfinite(torch.tensor(float(value))):
            raise RuntimeError(
                f"Non-finite scalar detected in {name} "
                f"(client={self.id}, round={round_idx}, epoch={local_epoch}, batch={batch_idx}, value={value})"
            )

    def _raise_if_nonfinite_grads(self, model, round_idx, local_epoch, batch_idx):
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            self._raise_if_nonfinite(f"grad[{name}]", param.grad, round_idx, local_epoch, batch_idx)


    def check_poison_timing(self, round):
        if round > self.args.cease_poison:
            self.train_dataset = utils.DatasetSplit(self.clean_backup_dataset, self.data_idxs)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                           num_workers=self.args.num_workers, pin_memory=False, drop_last=True)
    
    # def local_train_splitfedv1(
    #     self,
    #     global_client_model,          # <-- ONLY client-side model (ClientNet)
    #     main_server,                  # <-- MainServer from step 2 (per-client server branches)
    #     round=None,
    #     neurotoxin_mask=None,
    # ):
    #     """
    #     SplitFedV1 (Option B) local training:
    #     - client trains ONLY the client-side model
    #     - server-side is trained inside main_server.train_on_smashed(self.id, smashed, labels)
    #     - per batch: client forward -> server train -> server returns dfx -> client backward(dfx)

    #     Returns:
    #     self.update: vector update of client-side params (same idea as before, but only for client model)
    #     """

    #     # Snapshot initial client-side params (for update computation / defenses)
    #     initial_client_params = parameters_to_vector(
    #         [global_client_model.state_dict()[name] for name in global_client_model.state_dict()]
    #     ).detach()
    #     server_model = main_server.server_models[self.id]
    #     server_opt   = main_server.server_opts[self.id]

    #     server_before_sd  = copy.deepcopy(server_model.state_dict())
    #     opt_before_state  = copy.deepcopy(server_opt.state_dict())
    #     # Malicious timing hook (keep your existing behavior)
    #     if self.id < self.args.num_corrupt:
    #         self.check_poison_timing(round)

    #     global_client_model.train()
    #     global_client_model.apply(freeze_bn)

    #     optimizer = torch.optim.SGD(
    #         global_client_model.parameters(),
    #         lr=self.args.client_lr * (self.args.lr_decay) ** round,
    #         weight_decay=self.args.wd,
    #         momentum=self.args.momentum,
    #     )

    #     last_batch_loss = None

    #     for local_epoch in range(self.args.local_ep):
    #         start = time.time()

    #         for i, (inputs, labels) in enumerate(self.train_loader):
    #             optimizer.zero_grad(set_to_none=True)

    #             inputs = inputs.to(device=self.args.device, non_blocking=True)
    #             labels = labels.to(device=self.args.device, non_blocking=True)

    #             # If you have any poisoning/backdoor, it should happen HERE (before client forward),
    #             # because client is the only one seeing raw inputs.
    #             # Example (pseudocode):
    #             # if self.is_malicious and self.should_poison(round, local_epoch, i):
    #             #     inputs, labels = self.poison_batch(inputs, labels)

    #             # 1) Client forward (smashed activations)
    #             fx = global_client_model(inputs)

    #             # 2) Create server-side leaf tensor (server needs requires_grad=True to compute dL/dfx)
    #             smashed = fx.detach().clone().requires_grad_(True)

    #             # 3) Server trains its per-client branch and returns gradient w.r.t smashed
    #             # NOTE: main_server internally selects server_models[self.id]
    #             dfx, loss_s, acc_s = main_server.train_on_smashed(self.id, smashed, labels)
    #             last_batch_loss = loss_s

    #             # 4) Client backward using returned gradient
    #             fx.backward(dfx)
    #             if self.id == 0 and local_epoch == 0 and i == 0:
    #                 p0 = next(global_client_model.parameters())
    #                 print("mean|client_grad|", p0.grad.abs().mean().item() if p0.grad is not None else -1)

    #             # p0 = next(global_client_model.parameters())
    #             # print("mean|client_grad|", p0.grad.abs().mean().item() if p0.grad is not None else -1)


    #             # Optional: keep your neurotoxin-style gradient masking, but now it applies ONLY
    #             # to CLIENT-SIDE params (because that's all the client updates).
    #             if self.id < self.args.num_corrupt and neurotoxin_mask is not None and len(neurotoxin_mask):
    #                 if self.args.attack == "neurotoxin":
    #                     for name, param in global_client_model.named_parameters():
    #                         if name in neurotoxin_mask and param.grad is not None:
    #                             param.grad.data = neurotoxin_mask[name].to(self.args.device) * param.grad.data

    #                 if self.args.attack == "r_neurotoxin":
    #                     for name, param in global_client_model.named_parameters():
    #                         if name in neurotoxin_mask and param.grad is not None:
    #                             m = neurotoxin_mask[name].to(self.args.device)
    #                             param.grad.data = (torch.ones_like(m) - m) * param.grad.data

    #             optimizer.step()

    #             # PGD projection in your original code was based on FULL model params.
    #             # In SplitFedV1, do it only on CLIENT-SIDE params (that's what you return/aggregate).
    #             if (
    #                 self.args.attack == "pgd"
    #                 and self.id < self.args.num_corrupt
    #                 and (i == len(self.train_loader) - 1)
    #             ):
    #                 # Use the same eps logic but computed on client-side vector
    #                 if self.args.data == "cifar10":
    #                     eps = torch.norm(initial_client_params) * 0.1
    #                 else:
    #                     eps = torch.norm(initial_client_params)

    #                 current_client_params = parameters_to_vector(
    #                     [global_client_model.state_dict()[name] for name in global_client_model.state_dict()]
    #                 ).detach()

    #                 norm_diff = torch.norm(current_client_params - initial_client_params)
    #                 print("clip before (client-part): ", norm_diff)

    #                 if norm_diff > eps:
    #                     w_proj_vec = (
    #                         eps * (current_client_params - initial_client_params) / norm_diff
    #                         + initial_client_params
    #                     )
    #                     print("clip after (client-part): ", torch.norm(w_proj_vec - initial_client_params))

    #                     # Apply projected vector back to the CLIENT model
    #                     new_state_dict = utils.vector_to_model_wo_load(w_proj_vec, global_client_model)
    #                     global_client_model.load_state_dict(new_state_dict)

    #         end = time.time()
    #         train_time = end - start
    #         print(
    #             "local epoch %d \t client: %d \t mal: %s \t loss: %.8f \t time: %.2f"
    #             % (local_epoch, self.id, str(self.is_malicious), float(last_batch_loss), train_time)
    #         )

    #     # # Compute client-side update vector (what FedServer will aggregate / what your defenses inspect)
    #     # with torch.no_grad():
    #     #     after_train = parameters_to_vector(
    #     #         [global_client_model.state_dict()[name] for name in global_client_model.state_dict()]
    #     #     ).detach()
    #     #     self.update = after_train - initial_client_params
    #     #     # print("||client_update||", torch.norm(self.update).item())

    #     #     return self.update, se
    #     with torch.no_grad():
    #         # --- client update (as before) ---
    #         after_train = parameters_to_vector(
    #             [global_client_model.state_dict()[name] for name in global_client_model.state_dict()]
    #         ).detach()
    #         self.update = after_train - initial_client_params
 
    #         # --- NEW: server update (delta) ---
    #         server_after_sd = server_model.state_dict()
    #         server_update_dict = {}
    #         for k in server_before_sd.keys():
    #             server_update_dict[k] = (server_after_sd[k].detach() - server_before_sd[k].detach())

    #         # --- NEW: restore server branch so updates don't propagate before aggregation/filtering ---
    #         server_model.load_state_dict(server_before_sd)
    #         server_opt.load_state_dict(opt_before_state)

    #         return self.update, server_update_dict

    def _build_shadow_server_base(self, global_client_model, reference_server_model):
        mode = getattr(self.args, "shadow_server_strategy", "copy")
        if mode == "copy":
            return copy.deepcopy(reference_server_model).to(self.args.device).eval()

        if self.shadow_server_model is not None:
            return copy.deepcopy(self.shadow_server_model).to(self.args.device).eval()

        sample_inputs, _ = next(iter(self.train_loader))
        sample_inputs = sample_inputs.to(self.args.device)

        with torch.no_grad():
            smashed = global_client_model(sample_inputs[:1])

        if smashed.dim() == 4:
            channels = smashed.size(1)
            surrogate = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(channels, self.args.num_target),
            )
        elif smashed.dim() == 2:
            surrogate = nn.Sequential(
                nn.Linear(smashed.size(1), self.args.num_target),
            )
        else:
            surrogate = nn.Sequential(
                nn.Flatten(),
                nn.Linear(smashed[0].numel(), self.args.num_target),
            )

        return surrogate.to(self.args.device).eval()

    def _ensure_shadow_server_surrogate(self, global_client_model, reference_server_model, sample_inputs):
        if self.shadow_server_model is not None:
            return self.shadow_server_model

        sample_inputs = sample_inputs.to(self.args.device)
        with torch.no_grad():
            smashed = global_client_model(sample_inputs[:1])

        if smashed.dim() == 4:
            channels = smashed.size(1)
            surrogate = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(channels, self.args.num_target),
            )
        elif smashed.dim() == 2:
            surrogate = nn.Sequential(
                nn.Linear(smashed.size(1), self.args.num_target),
            )
        else:
            surrogate = nn.Sequential(
                nn.Flatten(),
                nn.Linear(smashed[0].numel(), self.args.num_target),
            )

        self.shadow_server_model = surrogate.to(self.args.device)
        self.shadow_server_opt = torch.optim.SGD(
            self.shadow_server_model.parameters(),
            lr=getattr(self.args, "surrogate_lr", 0.01),
            momentum=getattr(self.args, "momentum", 0.0),
            weight_decay=getattr(self.args, "wd", 0.0),
        )
        return self.shadow_server_model

    def _update_shadow_server_from_feedback(self, global_client_model, reference_server_model, inputs, labels, observed_dfx):
        if getattr(self.args, "shadow_server_strategy", "copy") != "surrogate":
            return
        if observed_dfx is None:
            return

        shadow_server = self._ensure_shadow_server_surrogate(
            global_client_model,
            reference_server_model,
            inputs,
        )
        shadow_server.train()

        update_steps = getattr(self.args, "surrogate_update_steps", 1)
        grad_weight = getattr(self.args, "surrogate_grad_weight", 1.0)
        cls_weight = getattr(self.args, "surrogate_cls_weight", 1.0)

        for _ in range(update_steps):
            smashed = global_client_model(inputs).detach().clone().requires_grad_(True)
            logits = shadow_server(smashed)
            cls_loss = nn.CrossEntropyLoss()(logits, labels)
            grad_surrogate = torch.autograd.grad(
                cls_loss,
                smashed,
                retain_graph=True,
                create_graph=True,
            )[0]
            grad_loss = torch.nn.functional.mse_loss(grad_surrogate, observed_dfx.detach())
            total_loss = cls_weight * cls_loss + grad_weight * grad_loss

            self.shadow_server_opt.zero_grad(set_to_none=True)
            total_loss.backward()
            self.shadow_server_opt.step()

        shadow_server.eval()

    @staticmethod
    def _flatten_grad_list(grad_list, params):
        flat = []
        for grad, param in zip(grad_list, params):
            if grad is None:
                flat.append(torch.zeros_like(param).view(-1))
            else:
                flat.append(grad.contiguous().view(-1))
        return torch.cat(flat)
    
    def get_adv_split_models(self, global_client_model, global_server_model, criterion):
        adv_pairs, adv_ws = [], []

        for _ in range(self.args.dm_adv_model_count):
            adv_client = copy.deepcopy(global_client_model).to(self.args.device)
            adv_server = copy.deepcopy(global_server_model).to(self.args.device)
            adv_client.train()
            adv_server.train()

            if self.args.model == "googlenet":
                adv_opt = torch.optim.Adam(
                    list(adv_client.parameters()) + list(adv_server.parameters()),
                    lr=0.01,
                )
            else:
                adv_opt = torch.optim.SGD(
                    list(adv_client.parameters()) + list(adv_server.parameters()),
                    lr=0.01,
                    momentum=0.9,
                )

            for _ in range(self.args.dm_adv_epochs):
                for inputs, _ in self.train_loader:
                    inputs = inputs.to(self.args.device)

                    poisoned = self.trigger.detach() * self.mask + (1 - self.mask) * inputs

                    with torch.no_grad():
                        clean_logits = global_server_model(global_client_model(inputs))
                        clean_labels = torch.argmax(clean_logits, dim=1)

                    adv_opt.zero_grad()
                    adv_logits = adv_server(adv_client(poisoned))
                    loss = criterion(adv_logits, clean_labels)
                    loss.backward()
                    adv_opt.step()

            adv_pairs.append((adv_client.eval(), adv_server.eval()))
            adv_ws.append(torch.tensor(1.0, device=self.args.device))

        return adv_pairs, adv_ws

    def local_train_splitfedv1(
    self,
    global_client_model,   # client-side model
    main_server,           # MainServer with per-client server branches
    round=None,
    neurotoxin_mask=None,   
    ):
        """
        SplitFedV1 local training (client trains client-part; server trains server-part per batch).
        Returns:
        - client_update_vec: 1D torch vector (client-part delta)
        - server_update_dict: dict(name -> tensor) (server-part delta for this client branch)
        """

        # --- Snapshot initial client-side params (for delta computation) ---
        initial_client_params = parameters_to_vector(
            [global_client_model.state_dict()[name] for name in global_client_model.state_dict()]
        ).detach()

        # --- Snapshot this client's server branch (for delta computation + restore) ---
        if not main_server.multiple_servers:
            server_model = main_server.server_models[0]
            server_opt = main_server.server_opts[0]
        else:
            server_model = main_server.server_models[self.id]
            server_opt = main_server.server_opts[self.id]

        # SplitFedV2 (main_server.sfl_variant == "v2"): the server-side model
        # is meant to be updated sequentially in place by every client in
        # turn, with no restore and no round-end aggregation (Thapa et al.
        # never FedAvg the server side in V2). SplitFedV1 (default) needs the
        # snapshot so it can restore the branch afterwards and hand only the
        # delta to aggregate_server_updates_round.
        is_sflv2 = getattr(main_server, "sfl_variant", "v1") == "v2"
        if not is_sflv2:
            server_before_sd = copy.deepcopy(server_model.state_dict())
            opt_before_state = copy.deepcopy(server_opt.state_dict())  # optional but keeps behavior identical

        # Malicious timing hook (keep existing)
        if self.id < self.args.num_corrupt:
            self.check_poison_timing(round)

        
        if self.id < self.args.num_corrupt and self.args.attack == "a3fl_opt":
            criterion = nn.CrossEntropyLoss()

            opt_inputs, _ = next(iter(self.train_loader))
            opt_inputs = opt_inputs.to(self.args.device)
            target_labels_opt = torch.full(
                (opt_inputs.size(0),),
                self.args.target_class,
                device=self.args.device,
                dtype=torch.long,
            )

            live_client = copy.deepcopy(global_client_model).to(self.args.device).eval()
            if not main_server.multiple_servers:
                s_model = main_server.server_models[0]
            else:
                s_model = main_server.server_models[self.id]
            live_server = self._build_shadow_server_base(
                live_client,
                s_model,
            )

            adv_pairs, adv_ws = self.get_adv_split_models(live_client, live_server, criterion)

            for _ in range(self.args.trigger_outter_epochs):
                self.trigger = self.trigger.detach().requires_grad_(True)
                p_input = self.trigger * self.mask + (1 - self.mask) * opt_inputs

                logits_live = live_server(live_client(p_input))

                loss_shadow = 0.0
                for (adv_client, adv_server), aw in zip(adv_pairs, adv_ws):
                    loss_shadow = loss_shadow + aw * criterion(
                        adv_server(adv_client(p_input)),
                        target_labels_opt,
                    )

                t_loss = criterion(logits_live, target_labels_opt) + \
                    (self.args.lambda_a3fl / len(adv_pairs)) * loss_shadow

                t_loss.backward()

                with torch.no_grad():
                    if self.trigger.grad is not None:
                        self.trigger -= self.args.trigger_lr * self.trigger.grad.sign()
                        self.trigger.clamp_(0, 1)

            del adv_pairs
            torch.cuda.empty_cache()


        
        # --- Client-side training setup ---
        global_client_model.train()
        global_client_model.apply(freeze_bn)

        if self.args.model == "googlenet":
            optimizer = torch.optim.Adam(
                global_client_model.parameters(),
                lr=self.args.client_lr * (self.args.lr_decay) ** round,
                weight_decay=self.args.wd,
            )
        else:
            optimizer = torch.optim.SGD(
                global_client_model.parameters(),
                lr=self.args.client_lr * (self.args.lr_decay) ** round,
                weight_decay=self.args.wd,
                momentum=self.args.momentum,
            )

        last_batch_loss = None
        clean_ref_iter = iter(self.clean_ref_loader) if self.clean_ref_loader is not None else None

        for local_epoch in range(self.args.local_ep):
            start = time.time()

            for i, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad(set_to_none=True)

                inputs = inputs.to(device=self.args.device, non_blocking=True)
                labels = labels.to(device=self.args.device, non_blocking=True)

                if self.id < self.args.num_corrupt and self.args.attack == "alignment":
                    if clean_ref_iter is None:
                        raise RuntimeError("alignment attack requires a clean reference loader for malicious clients")
                    try:
                        clean_inputs, _ = next(clean_ref_iter)
                    except StopIteration:
                        clean_ref_iter = iter(self.clean_ref_loader)
                        clean_inputs, _ = next(clean_ref_iter)

                    clean_inputs = clean_inputs.to(device=self.args.device, non_blocking=True)
                    target_labels = torch.full_like(labels, self.args.target_class)

                    client_params = [p for p in global_client_model.parameters() if p.requires_grad]

                    benign_server = copy.deepcopy(server_model).to(self.args.device).eval()
                    benign_logits = benign_server(global_client_model(clean_inputs))
                    benign_targets = benign_logits.detach().argmax(dim=1)
                    benign_loss = nn.CrossEntropyLoss()(benign_logits, benign_targets)
                    benign_grads = torch.autograd.grad(
                        benign_loss,
                        client_params,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                    g_benign = self._flatten_grad_list(benign_grads, client_params).detach()
                    self._raise_if_nonfinite("alignment/benign_loss", benign_loss, round, local_epoch, i)
                    self._raise_if_nonfinite("alignment/g_benign", g_benign, round, local_epoch, i)
                    del benign_server

                    current_server = copy.deepcopy(server_model).to(self.args.device).eval()
                    shadow_server = self._build_shadow_server_base(global_client_model, server_model)
                    poison_logits = current_server(global_client_model(inputs))
                    shadow_logits = shadow_server(global_client_model(inputs))
                    loss_poison = nn.CrossEntropyLoss()(poison_logits, target_labels) + \
                        self.args.lambda_a3fl * nn.CrossEntropyLoss()(shadow_logits, target_labels)

                    poison_grads = torch.autograd.grad(
                        loss_poison,
                        client_params,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True,
                    )
                    g_poison = self._flatten_grad_list(poison_grads, client_params)
                    self._raise_if_nonfinite("alignment/loss_poison", loss_poison, round, local_epoch, i)
                    self._raise_if_nonfinite("alignment/g_poison", g_poison, round, local_epoch, i)
                    cos_sim = torch.nn.functional.cosine_similarity(
                        g_poison.unsqueeze(0),
                        g_benign.unsqueeze(0),
                        dim=1,
                        eps=1e-12,
                    ).squeeze(0)
                    loss_dir = 1 - cos_sim
                    mismatch = (g_poison * g_benign < 0).float()
                    loss_sign = torch.sum(torch.abs(g_poison) * mismatch)
                    self._raise_if_nonfinite("alignment/cos_sim", cos_sim, round, local_epoch, i)
                    self._raise_if_nonfinite("alignment/loss_dir", loss_dir, round, local_epoch, i)
                    self._raise_if_nonfinite("alignment/loss_sign", loss_sign, round, local_epoch, i)
                    total_loss = loss_poison + \
                        self.args.alignment_alpha * loss_dir + \
                        self.args.alignment_beta * loss_sign
                    self._raise_if_nonfinite("alignment/total_loss", total_loss, round, local_epoch, i)

                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    self._raise_if_nonfinite_grads(global_client_model, round, local_epoch, i)
                    optimizer.step()

                    with torch.no_grad():
                        smashed = global_client_model(inputs).detach().clone().requires_grad_(True)
                    dfx, loss_s, acc_s = main_server.train_on_smashed(self.id, smashed, target_labels)
                    self._update_shadow_server_from_feedback(
                        global_client_model,
                        server_model,
                        inputs,
                        target_labels,
                        dfx,
                    )
                    last_batch_loss = total_loss.detach()
                    continue

                if self.id < self.args.num_corrupt and self.args.attack == "a3fl_opt":
                    inputs = self.trigger.detach() * self.mask + (1 - self.mask) * inputs
                    labels = torch.full_like(labels, self.args.target_class)


                # (Optional) poison inputs/labels here if needed

                # 1) Client forward -> smashed activations
                fx = global_client_model(inputs)
                # print(fx.abs().mean(), fx.std(), fx.norm())
                # After fx = global_client_model(inputs)
                if self.id < self.args.num_corrupt and i == 0:
                    # Compare the activations of poisoned samples vs potentially clean ones
                    poison_mask = (labels == self.args.target_class)
                    clean_mask = (labels != self.args.target_class)
                    
                    if poison_mask.any() and clean_mask.any():
                        mean_poison_act = fx[poison_mask].mean().item()
                        mean_clean_act = fx[clean_mask].mean().item()
                        print(f"Smashed Data Mean - Poisoned: {mean_poison_act:.4f} | Clean: {mean_clean_act:.4f}")
                        print(f"Signal Difference: {abs(mean_poison_act - mean_clean_act):.6f}")

                # 2) Server consumes smashed as leaf tensor
                smashed = fx.detach().clone().requires_grad_(True)

                # 3) Server trains its branch and returns dL/dfx
                dfx, loss_s, acc_s = main_server.train_on_smashed(self.id, smashed, labels)
                dfx = dfx.to(fx.device)
                last_batch_loss = loss_s
                self._update_shadow_server_from_feedback(
                    global_client_model,
                    server_model,
                    inputs,
                    labels,
                    dfx,
                )
                # After dfx, loss_s, acc_s = main_server.train_on_smashed(...)
                if self.id < self.args.num_corrupt and i == 0:
                    grad_norm = dfx.norm().item()
                    print(f"Server Gradient (dfx) Norm: {grad_norm:.6f}")
                    if grad_norm < 1e-7:
                        print("WARNING: Server feedback is vanishing! The server part is not learning the trigger.")

                # 4) Client backward using server-provided gradient
                fx.backward(dfx)

                # (Optional) debug
                if self.id == 0 and local_epoch == 0 and i == 0:
                    p0 = next(global_client_model.parameters())
                    print("mean|client_grad|", p0.grad.abs().mean().item() if p0.grad is not None else -1)

                # --- Optional neurotoxin-style masking on CLIENT grads only ---
                if self.id < self.args.num_corrupt and neurotoxin_mask is not None and len(neurotoxin_mask):
                    if self.args.attack == "neurotoxin":
                        for name, param in global_client_model.named_parameters():
                            if name in neurotoxin_mask and param.grad is not None:
                                param.grad.data = neurotoxin_mask[name].to(self.args.device) * param.grad.data

                    if self.args.attack == "r_neurotoxin":
                        for name, param in global_client_model.named_parameters():
                            if name in neurotoxin_mask and param.grad is not None:
                                m = neurotoxin_mask[name].to(self.args.device)
                                param.grad.data = (torch.ones_like(m) - m) * param.grad.data

                # 5) Client optimizer step
                optimizer.step()

                # --- PGD projection on CLIENT params only (if enabled) ---
                if (
                    self.args.attack == "pgd"
                    and self.id < self.args.num_corrupt
                    and (i == len(self.train_loader) - 1)
                ):
                    eps = torch.norm(initial_client_params) * 0.1 if self.args.data == "cifar10" else torch.norm(initial_client_params)

                    current_client_params = parameters_to_vector(
                        [global_client_model.state_dict()[name] for name in global_client_model.state_dict()]
                    ).detach()

                    norm_diff = torch.norm(current_client_params - initial_client_params)
                    print("clip before (client-part): ", norm_diff)

                    if norm_diff > eps:
                        w_proj_vec = (
                            eps * (current_client_params - initial_client_params) / norm_diff
                            + initial_client_params
                        )
                        print("clip after (client-part): ", torch.norm(w_proj_vec - initial_client_params))

                        new_state_dict = utils.vector_to_model_wo_load(w_proj_vec, global_client_model)
                        global_client_model.load_state_dict(new_state_dict)

            train_time = time.time() - start
            print(
                "local epoch %d \t client: %d \t mal: %s \t loss: %.8f \t time: %.2f"
                % (local_epoch, self.id, str(self.is_malicious), float(last_batch_loss), train_time)
            )

        # --- Build deltas + restore server branch ---
        with torch.no_grad():
            # Client delta vector
            after_train = parameters_to_vector(
                [global_client_model.state_dict()[name] for name in global_client_model.state_dict()]
            ).detach()
            client_update_vec = after_train - initial_client_params
            self.update = client_update_vec
            
            
            
            # --- MODEL REPLACEMENT (Scaling) ---
            # Default scale is 1.0 (Baseline)
            # client_update_vec = after_train - initial_client_params
            scale = 1.0
            if self.id < self.args.num_corrupt and self.args.attack_mode == "replacement":
                # scale = Total Agents / (Agents per round * lr)
                # For simplicity in many SFL papers, we use N / m
                scale = self.args.num_agents / (self.args.num_agents * self.args.agent_frac)
            
                # Apply scaling to the delta
                client_update_vec = (after_train - initial_client_params) * scale
                self.update = client_update_vec
            # --- PGD CONSTRAINT ---
            if self.id < self.args.num_corrupt and self.args.attack_mode == "pgd":
                # Calculate norm of the (potentially scaled) update
                eps = torch.norm(initial_client_params) * self.args.pgd_eps
                actual_norm = torch.norm(client_update_vec)
                
                if actual_norm > eps:
                    client_update_vec = (client_update_vec / actual_norm) * eps
            
            
                # client_update_vec = after_train - initial_client_params
                self.update = client_update_vec

            if is_sflv2:
                # SplitFedV2: leave the server-side model exactly as this
                # client's training left it -- no restore, no delta to
                # report (aggregate_server_updates_round is a no-op in this
                # mode too, so an empty dict is never actually consumed).
                server_update_dict = {}
            else:
                # Server delta dict (after - before)
                server_after_sd = {k: v.detach().clone() for k, v in server_model.state_dict().items()}
                server_update_dict = {}
                for k in server_before_sd.keys():
                    server_update_dict[k] = (server_after_sd[k] - server_before_sd[k].detach()).to(self.args.device)

                # Restore server state so it doesn't propagate before robust aggregation
                server_model.load_state_dict(server_before_sd)
                server_opt.load_state_dict(opt_before_state)  # optional, kept for identical behavior
            # After dfx, loss_s, acc_s = main_server.train_on_smashed(...)
            if self.id < self.args.num_corrupt and i == 0:
                grad_norm = dfx.norm().item()
                print(f"Server Gradient (dfx) Norm: {grad_norm:.6f}")
                if grad_norm < 1e-7:
                    print("WARNING: Server feedback is vanishing! The server part is not learning the trigger.")
        return client_update_vec, server_update_dict

    def local_train(self, global_model, criterion, round=None, neurotoxin_mask=None):

        # print(len(self.train_dataset))
        """ Do a local training over the received global model, return the update """
        # start = time.time()
        initial_global_model_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.id < self.args.num_corrupt:
            self.check_poison_timing(round)
        global_model.train()
        if self.args.model == "googlenet":
            optimizer = torch.optim.Adam(
                global_model.parameters(),
                lr=self.args.client_lr * (self.args.lr_decay) ** round,
                weight_decay=self.args.wd,
            )
        else:
            optimizer = torch.optim.SGD(global_model.parameters(),
                                        lr=self.args.client_lr * (self.args.lr_decay) ** round,
                                        weight_decay=self.args.wd, momentum=self.args.momentum)

        regular_loss = 0.0
        for local_epoch in range(self.args.local_ep):
            start = time.time()
            old_gradient = {}
            old_gradient_mine = {}
            old_params = {}
            for i, (inputs, labels) in enumerate(self.train_loader):
                # if i == 0 and self.is_malicious:
                #     save_image(torch.cat([inputs[labels == self.args.target_class][:10]]), '%s_image.png' % self.id, normalize=True, nrow=10)
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
                                 labels.to(device=self.args.device, non_blocking=True)
                outputs = global_model(inputs)
                # outputs = outputs[:, :]
                minibatch_loss = criterion(outputs, labels)
                # print(minibatch_loss)
                minibatch_loss.backward()
                if self.args.attack == "neurotoxin" and len(neurotoxin_mask) and self.id < self.args.num_corrupt:
                    for name, param in global_model.named_parameters():
                        param.grad.data = neurotoxin_mask[name].to(self.args.device) * param.grad.data
                if self.args.attack == "r_neurotoxin" and len(neurotoxin_mask) and self.id < self.args.num_corrupt:
                    for name, param in global_model.named_parameters():
                        param.grad.data = (torch.ones_like(neurotoxin_mask[name].to(self.args.device))-neurotoxin_mask[name].to(self.args.device) ) * param.grad.data
                optimizer.step()

                if self.args.attack == 'pgd' and self.id < self.args.num_corrupt and (i == len(self.train_loader) - 1):
                    if self.args.data == 'cifar10':
                        eps = torch.norm(initial_global_model_params) * 0.1
                    else:
                        eps = torch.norm(initial_global_model_params)

                    current_local_model_params = parameters_to_vector([global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
                    norm_diff = torch.norm(current_local_model_params - initial_global_model_params)
                    print('clip before: ', norm_diff)
                    if norm_diff > eps:
                        w_proj_vec = eps * (current_local_model_params - initial_global_model_params) / norm_diff + initial_global_model_params

                        print('clip after: ', torch.norm(w_proj_vec - initial_global_model_params))

                        new_state_dict = utils.vector_to_model_wo_load(w_proj_vec, global_model)    
                        global_model.load_state_dict(new_state_dict)

            end = time.time()
            train_time = end - start
            print("local epoch %d \t client: %d \t mal: %s \t loss: %.8f \t time: %.2f" % (local_epoch, self.id, str(self.is_malicious),
                                                                     minibatch_loss, train_time))

        with torch.no_grad():
            after_train = parameters_to_vector(
                [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
            self.update = after_train - initial_global_model_params

            return self.update
class FullSplitForEval(nn.Module):
    def __init__(self, client, server):
        super().__init__()
        self.client = client
        self.server = server
    def forward(self, x):
        fx = self.client(x)
        return self.server(fx)
