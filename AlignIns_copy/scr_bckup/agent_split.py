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
        if self.args.data != "tinyimagenet":
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
        server_model = main_server.server_models[self.id]
        server_opt = main_server.server_opts[self.id]
        server_before_sd = copy.deepcopy(server_model.state_dict())
        opt_before_state = copy.deepcopy(server_opt.state_dict())  # optional but keeps behavior identical

        # Malicious timing hook (keep existing)
        if self.id < self.args.num_corrupt:
            self.check_poison_timing(round)

        # --- Client-side training setup ---
        global_client_model.train()
        global_client_model.apply(freeze_bn)

        optimizer = torch.optim.SGD(
            global_client_model.parameters(),
            lr=self.args.client_lr * (self.args.lr_decay) ** round,
            weight_decay=self.args.wd,
            momentum=self.args.momentum,
        )

        last_batch_loss = None

        for local_epoch in range(self.args.local_ep):
            start = time.time()

            for i, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad(set_to_none=True)

                inputs = inputs.to(device=self.args.device, non_blocking=True)
                labels = labels.to(device=self.args.device, non_blocking=True)

                # (Optional) poison inputs/labels here if needed

                # 1) Client forward -> smashed activations
                fx = global_client_model(inputs)

                # 2) Server consumes smashed as leaf tensor
                smashed = fx.detach().clone().requires_grad_(True)

                # 3) Server trains its branch and returns dL/dfx
                dfx, loss_s, acc_s = main_server.train_on_smashed(self.id, smashed, labels)
                last_batch_loss = loss_s

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

            # Server delta dict (after - before)
            server_after_sd = {k: v.detach().clone() for k, v in server_model.state_dict().items()}
            server_update_dict = {}
            for k in server_before_sd.keys():
                server_update_dict[k] = (server_after_sd[k] - server_before_sd[k].detach()).to(self.args.device)

            # Restore server state so it doesn't propagate before robust aggregation
            server_model.load_state_dict(server_before_sd)
            server_opt.load_state_dict(opt_before_state)  # optional, kept for identical behavior

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
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr * (self.args.lr_decay) ** round,
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