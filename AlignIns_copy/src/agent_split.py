import copy
import time
import logging
import torch
import torch.nn as nn
import utils
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
import models

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
        self.proxy_server = None #for the version 2, when client doesn't know which model is used and cut layer
        self.smashed_memory = []
        
        
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
        if self.id < args.num_corrupt: #this assumes that first few clients are mal TODO allow different approaches also
            sample_img, _ = self.train_dataset[0]
            self.trigger_image = torch.zeros_like(sample_img).to(self.args.device)
            self.trigger_image.requires_grad = True

    def check_poison_timing(self, round):
        if round > self.args.cease_poison:
            self.train_dataset = utils.DatasetSplit(self.clean_backup_dataset, self.data_idxs)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                           num_workers=self.args.num_workers, pin_memory=False, drop_last=True)
    
    # def search_a3fl_trigger(self, global_client_model):  # version with not knowing cut layer
    #     if len(self.smashed_memory) < 10:
    #         return 

    #     if self.proxy_server is None:
    #         # Get shape: [Batch, Channels, Height, Width]
    #         sample_smashed = self.smashed_memory[0][0]
    #         channels = sample_smashed.shape[1] 
            
    #         # Tiny Proxy: Pooling -> Flatten -> Linear
    #         self.proxy_server = nn.Sequential(
    #             nn.AdaptiveAvgPool2d((1, 1)), # Shrinks HxW to 1x1, keeps Channels
    #             nn.Flatten(),                 # Turns [B, C, 1, 1] into [B, C]
    #             nn.Linear(channels, 32),
    #             nn.ReLU(),
    #             nn.Linear(32, channels)       # Predicts one value per channel
    #         ).to(self.args.device)
        
    #     # Use SGD to save memory (Adam uses 3x more VRAM)
    #     proxy_opt = torch.optim.SGD(self.proxy_server.parameters(), lr=1e-2)
        
    #     # Step 1: Train Proxy to predict Channel-Averages of Gradients
    #     self.proxy_server.train()
    #     for _ in range(3):
    #         for sm, real_dfx in self.smashed_memory:
    #             # DON'T use .view(-1) here, keep the 4D shape
    #             sm_gpu = sm.to(self.args.device) 
    #             dfx_gpu = real_dfx.to(self.args.device)
                
    #             # We predict the average gradient per channel
    #             pred_dfx_channels = self.proxy_server(sm_gpu) 
                
    #             # Target: Calculate actual average gradient per channel from the server feedback
    #             # real_dfx shape: [B, C, H, W] -> mean over H, W (dims 2, 3)
    #             target_dfx_channels = torch.mean(dfx_gpu, dim=(2, 3)) 
                
    #             loss = torch.nn.functional.mse_loss(pred_dfx_channels, target_dfx_channels)
                
    #             proxy_opt.zero_grad()
    #             loss.backward()
    #             proxy_opt.step()
        
    #     # --- Pixel Optimization ---
    #     self.proxy_server.eval()
    #     global_client_model.eval()
        
    #     for _ in range(3):
    #         for i, (inputs, _) in enumerate(self.train_loader):
    #             if i > 2: break
    #             inputs = inputs.to(self.args.device)
    #             poison_inputs = torch.clamp(inputs + self.trigger_image, 0, 1)
                
    #             fx = global_client_model(poison_inputs)
                
    #             with torch.no_grad():
    #                 # Predict channel importance, then expand back to full fx shape
    #                 pred_channels = self.proxy_server(fx).view(fx.shape[0], fx.shape[1], 1, 1)
    #                 predicted_dfx = pred_channels.expand_as(fx)
                
    #             fx.backward(predicted_dfx)
                
    #             self.trigger_image.data -= self.args.trigger_lr * self.trigger_image.grad.sign()
    #             self.trigger_image.data.clamp_(0, 1)
    #             self.trigger_image.grad.zero_()

    #     torch.cuda.empty_cache()


    def search_a3fl_trigger(self, global_client_model):
        """
        Architecture-Aware Proxy Search (Weights Unknown).
        The proxy learns to predict server gradients (df/dx) from smashed data.
        """
        if len(self.smashed_memory) < 10:
            return 

        # 1. Initialize the architecture (Known by assumption)
        # But weights are random/standard initialization
        if self.proxy_server is None:
            _, self.proxy_server, _ = models.get_split_models(self.args.data, self.args)
            self.proxy_server = self.proxy_server.to(self.args.device)
        
        proxy_opt = torch.optim.SGD(self.proxy_server.parameters(), lr=1e-3)
        
        # 2. TRAINING THE PROXY (Learning the Server's Behavior)
        # We treat this as a regression task: Predict the Gradient Tensor
        self.proxy_server.train()
        for _ in range(5): # Small inner loop to catch up to server state
            for sm, real_dfx in self.smashed_memory:
                sm_gpu = sm.to(self.args.device)
                real_dfx_gpu = real_dfx.to(self.args.device)
                
                # Forward pass on our local "shadow" architecture
                # We want to see what gradients OUR local model generates
                sm_gpu.requires_grad_(True)
                outputs = self.proxy_server(sm_gpu)
                
                # We use a dummy loss to generate a gradient we can compare
                # In A3FL, we assume the attacker's target class is the goal
                target_labels = torch.full((sm_gpu.size(0),), self.args.target_class).to(self.args.device)
                dummy_loss = nn.CrossEntropyLoss()(outputs, target_labels)
                
                # pred_dfx is what OUR architecture thinks the gradient should be
                pred_dfx = torch.autograd.grad(dummy_loss, sm_gpu, create_graph=True)[0]
                
                # CRITICAL STEP: We update the proxy weights so that 
                # its predicted gradient (pred_dfx) matches the server's real gradient (real_dfx)
                proxy_loss = torch.nn.functional.mse_loss(pred_dfx, real_dfx_gpu)
                
                proxy_opt.zero_grad()
                proxy_loss.backward()
                proxy_opt.step()

        # 3. TRIGGER OPTIMIZATION (Using the trained Proxy)
        self.proxy_server.eval()
        global_client_model.eval()
        
        for _ in range(self.args.trigger_outter_epochs):
            for i, (inputs, _) in enumerate(self.train_loader):
                if i > 2: break
                inputs = inputs.to(self.args.device)
                
                # Apply current trigger
                poison_inputs = torch.clamp(inputs + self.trigger_image, 0, 1)
                
                # Forward through client layers
                fx = global_client_model(poison_inputs)
                
                # Use the Proxy to "hallucinate" what the server would say
                # We don't need a full forward pass through the proxy here, 
                # we just need it to provide the dfx (gradients) for this fx.
                fx_for_grad = fx.detach().clone().requires_grad_(True)
                proxy_out = self.proxy_server(fx_for_grad)
                
                target_labels = torch.full((fx.size(0),), self.args.target_class).to(self.args.device)
                proxy_loss = nn.CrossEntropyLoss()(proxy_out, target_labels)
                
                # Get the "hallucinated" gradient
                hallucinated_dfx = torch.autograd.grad(proxy_loss, fx_for_grad)[0]
                
                # Backpropagate that hallucinated gradient to the trigger pixels
                fx.backward(hallucinated_dfx)
                
                # Update pixels via Sign-SGD
                self.trigger_image.data -= self.args.trigger_lr * self.trigger_image.grad.sign()
                self.trigger_image.data.clamp_(0, 1)
                self.trigger_image.grad.zero_()

        torch.cuda.empty_cache()

    # def local_train_splitfedv1(
    #     self,
    #     global_client_model,   # client-side model
    #     main_server,           # MainServer with per-client server branches
    #     round=None,
    #     neurotoxin_mask=None,   
    #     ):
    #     """
    #         SplitFedV1 local training (client trains client-part; server trains server-part per batch).
    #         Returns:
    #         - client_update_vec: 1D torch vector (client-part delta)
    #         - server_update_dict: dict(name -> tensor) (server-part delta for this client branch)
    #     """

    #     # --- Snapshot initial client-side params (for delta computation) ---
    #     initial_client_params = parameters_to_vector(
    #         [global_client_model.state_dict()[name] for name in global_client_model.state_dict()]
    #     ).detach()
    #     logging.info(f" [Client {self.id}] Starting Split-Training. Communicating with MainServer...")
    #     # --- Snapshot this client's server branch (for delta computation + restore) ---
    #     server_model = main_server.server_models[self.id]
    #     server_opt = main_server.server_opts[self.id]
    #     server_before_sd = copy.deepcopy(server_model.state_dict())
    #     opt_before_state = copy.deepcopy(server_opt.state_dict())  # optional but keeps behavior identical


    #     if self.id < self.args.num_corrupt and self.args.attack == "a3fl":
    #         # Only search if we have enough server 'history' to learn from
    #         if len(self.smashed_memory) >= 10: 
    #             logging.info(f" [Client {self.id}] Running Black-Box A3FL search...")
    #             self.search_a3fl_trigger(global_client_model)
    #     # Malicious timing hook (keep existing)
    #     if self.id < self.args.num_corrupt:
    #         self.check_poison_timing(round)

    #     # --- Client-side training setup ---
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

    #             # (Optional) poison inputs/labels here if needed

    #             # 1) Client forward -> smashed activations
    #             fx = global_client_model(inputs)

    #             # 2) Server consumes smashed as leaf tensor
    #             smashed = fx.detach().clone().requires_grad_(True)

    #             # 3) Server trains its branch and returns dL/dfx
    #             dfx, loss_s, acc_s = main_server.train_on_smashed(self.id, smashed, labels)
    #             last_batch_loss = loss_s
    #             if self.args.debug and i == 0:
    #                 # If dfx is always pushing one way, the server is the 'attacker'
    #                 logging.info(f" [Client {self.id}] Server Gradient (dfx) Max: {dfx.max().item():.4f} | Min: {dfx.min().item():.4f}")
    #             # 4) Client backward using server-provided gradient
    #             fx.backward(dfx)
    #             # --- NEW: Capture data for Black-Box Proxy ---
    #             if self.id < self.args.num_corrupt:
    #                 # Store smashed data and the gradient the server returned
    #                 self.smashed_memory.append((smashed.detach().clone(), dfx.detach().clone()))
    #                 # Keep memory manageable (e.g., last 200 batches)
    #                 if len(self.smashed_memory) > 200:
    #                     self.smashed_memory.pop(0)
    #             # (Optional) debug
    #             if self.id == 0 and local_epoch == 0 and i == 0:
    #                 p0 = next(global_client_model.parameters())
                
    #             # --- Optional neurotoxin-style masking on CLIENT grads only ---
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

    #             # 5) Client optimizer step
    #             optimizer.step()

    #             # --- PGD projection on CLIENT params only (if enabled) ---
    #             if (
    #                 self.args.attack == "pgd"
    #                 and self.id < self.args.num_corrupt
    #                 and (i == len(self.train_loader) - 1)
    #             ):
    #                 eps = torch.norm(initial_client_params) * 0.1 if self.args.data == "cifar10" else torch.norm(initial_client_params)

    #                 current_client_params = parameters_to_vector(
    #                     [global_client_model.state_dict()[name] for name in global_client_model.state_dict()]
    #                 ).detach()

    #                 norm_diff = torch.norm(current_client_params - initial_client_params)
    #                 # logging.info("clip before (client-part): ", norm_diff)

    #                 if norm_diff > eps:
    #                     w_proj_vec = (
    #                         eps * (current_client_params - initial_client_params) / norm_diff
    #                         + initial_client_params
    #                     )
    #                     # v("clip after (client-part): ", torch.norm(w_proj_vec - initial_client_params))

    #                     new_state_dict = utils.vector_to_model_wo_load(w_proj_vec, global_client_model)
    #                     global_client_model.load_state_dict(new_state_dict)

    #         train_time = time.time() - start
    #         logging.info(
    #             "local epoch %d \t client: %d \t mal: %s \t loss: %.8f \t time: %.2f"
    #             % (local_epoch, self.id, str(self.is_malicious), float(last_batch_loss), train_time)
    #         )
    #     logging.info(f" [Client {self.id}] Epoch {local_epoch} complete. Final Batch Loss: {float(last_batch_loss):.4f}")
    #     # --- Build deltas + restore server branch ---
    #     with torch.no_grad():
    #         # Client delta vector
    #         after_train = parameters_to_vector(
    #             [global_client_model.state_dict()[name] for name in global_client_model.state_dict()]
    #         ).detach()
    #         client_update_vec = after_train - initial_client_params
    #         self.update = client_update_vec

    #         # Server delta dict (after - before)
    #         server_after_sd = {k: v.detach().clone() for k, v in server_model.state_dict().items()}
    #         server_update_dict = {}
    #         for k in server_before_sd.keys():
    #             server_update_dict[k] = (server_after_sd[k] - server_before_sd[k].detach()).to(self.args.device)
    #         logging.info(f" [Client {self.id}] Local training finished. Sending bottom-model update to FedServer.")
    #         # Restore server state so it doesn't propagate before robust aggregation
    #         server_model.load_state_dict(server_before_sd)
    #         server_opt.load_state_dict(opt_before_state)  # optional, kept for identical behavior

    #     return client_update_vec, server_update_dict
    def local_train_splitfedv1(
        self,
        global_client_model,   # client-side model
        main_server,           # MainServer with per-client server branches
        round=None,
        neurotoxin_mask=None,   
    ):
        """
        SplitFedV1 local training with integrated A3FL Black-Box support.
        Preserves original BadNet/DBA/Neurotoxin/PGD logic.
        """

        # --- 1. Snapshot Parameters ---
        initial_client_params = parameters_to_vector(
            [global_client_model.state_dict()[name] for name in global_client_model.state_dict()]
        ).detach()
        
        server_model = main_server.server_models[self.id]
        server_opt = main_server.server_opts[self.id]
        server_before_sd = copy.deepcopy(server_model.state_dict())
        opt_before_state = copy.deepcopy(server_opt.state_dict())

        # # --- 2. Offline Attack Optimization (A3FL Only) ---
        # if self.id < self.args.num_corrupt and self.args.attack == "a3fl":
        #     # Only optimize if we have gathered enough 'Smashed Memory' from previous rounds
        #     if len(self.smashed_memory) >= 10: 
        #         logging.info(f" [Client {self.id}] Running Black-Box A3FL trigger search...")
        #         self.search_a3fl_trigger(global_client_model)

        # --- 2. Offline Attack Optimization (A3FL White-Box) ---
        if self.id < self.args.num_corrupt and self.args.attack == "a3fl":
            if len(self.smashed_memory) >= 10: 
                logging.info(f" [Client {self.id}] Running White-Box A3FL trigger search (Architecture Aware)...")
                self.search_a3fl_trigger(global_client_model)

        # Malicious timing hook
        if self.id < self.args.num_corrupt:
            self.check_poison_timing(round)

        # --- 3. Training Setup ---
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

                # --- 4. Attack-Specific Data Modification ---
                if self.id < self.args.num_corrupt:
                    if self.args.attack == "a3fl":
                        # Apply the dynamically optimized trigger pixels
                        inputs = torch.clamp(inputs + self.trigger_image, 0, 1)
                        labels = torch.full_like(labels, self.args.target_class)
                    
                    elif self.args.attack in ["badnet", "DBA", "neurotoxin"]:
                        # For these, the trigger is likely already in 'inputs' via utils.poison_dataset
                        # We simply ensure the labels are set to the target class
                        labels = torch.full_like(labels, self.args.target_class)

                # --- 5. Forward Pass & Handshake ---
                # 1) Client forward -> smashed activations
                fx = global_client_model(inputs)

                # 2) Server consumes smashed as leaf tensor
                smashed = fx.detach().clone().requires_grad_(True)

                # 3) Server trains its branch and returns dL/dfx
                dfx, loss_s, acc_s = main_server.train_on_smashed(self.id, smashed, labels)
                last_batch_loss = loss_s
                
                # --- 6. Black-Box Data Collection (For A3FL) ---
                if self.id < self.args.num_corrupt:
                    # Record this interaction to train the Proxy in the next round
                    self.smashed_memory.append((smashed.detach().cpu(), dfx.detach().cpu()))
                    if len(self.smashed_memory) > 20:
                        self.smashed_memory.pop(0)

                if self.args.debug and i == 0:
                    logging.info(f" [Client {self.id}] Server Gradient (dfx) Max: {dfx.max().item():.4f}")

                # --- 7. Backward Pass & Gradient Masking ---
                fx.backward(dfx)

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

                # --- 8. Optimization Step ---
                optimizer.step()

                # --- 9. PGD Projection (Post-Training Constraint) ---
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
                    if norm_diff > eps:
                        w_proj_vec = (eps * (current_client_params - initial_client_params) / norm_diff) + initial_client_params
                        new_state_dict = utils.vector_to_model_wo_load(w_proj_vec, global_client_model)
                        global_client_model.load_state_dict(new_state_dict)

            train_time = time.time() - start
            logging.info(f"local epoch {local_epoch} \t client: {self.id} \t mal: {self.is_malicious} \t loss: {float(last_batch_loss):.8f}")

        # --- 10. Build Deltas & Restore Server Branch ---
        with torch.no_grad():
            after_train = parameters_to_vector(
                [global_client_model.state_dict()[name] for name in global_client_model.state_dict()]
            ).detach()
            client_update_vec = after_train - initial_client_params
            
            server_after_sd = {k: v.detach().clone() for k, v in server_model.state_dict().items()}
            server_update_dict = {k: (server_after_sd[k] - server_before_sd[k].detach()).to(self.args.device) for k in server_before_sd.keys()}
            
            # Vital for SplitFed: Restore server state for the aggregator
            server_model.load_state_dict(server_before_sd)
            server_opt.load_state_dict(opt_before_state)

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
                    logging.info('clip before: ', norm_diff)
                    if norm_diff > eps:
                        w_proj_vec = eps * (current_local_model_params - initial_global_model_params) / norm_diff + initial_global_model_params

                        logging.info('clip after: ', torch.norm(w_proj_vec - initial_global_model_params))

                        new_state_dict = utils.vector_to_model_wo_load(w_proj_vec, global_model)    
                        global_model.load_state_dict(new_state_dict)

            end = time.time()
            train_time = end - start
            logging.info("local epoch %d \t client: %d \t mal: %s \t loss: %.8f \t time: %.2f" % (local_epoch, self.id, str(self.is_malicious),
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
