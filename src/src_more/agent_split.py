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

        if self.args.data not in ("gtsrb", "ets"):
            self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)

            # for backdoor attack, agent poisons his local dataset
            if self.id < args.num_corrupt and self.args.attack != 'non':

                self.clean_backup_dataset = copy.deepcopy(train_dataset)
                self.data_idxs = data_idxs
                utils.poison_dataset(train_dataset, args, data_idxs, agent_idx=self.id)
            elif self.id < args.num_corrupt and self.args.attack != 'non':
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
            if self.args.data not in ("gtsrb", "ets"):
                clean_ref_dataset = utils.DatasetSplit(self.clean_backup_dataset, self.data_idxs)
                self.clean_ref_loader = DataLoader(
                    clean_ref_dataset,
                    batch_size=self.args.bs,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=False,
                    drop_last=True,
                )
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

        # Snapshot so we can restore the branch afterwards and hand only the
        # delta to aggregate_server_updates_round.
        server_before_sd = copy.deepcopy(server_model.state_dict())
        opt_before_state = copy.deepcopy(server_opt.state_dict())  # optional but keeps behavior identical

        # Malicious timing hook (keep existing)
        if self.id < self.args.num_corrupt:
            self.check_poison_timing(round)
        
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
