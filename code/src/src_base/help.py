def local_train_splitfedv1(
        self,
        global_client_model,
        main_server,
        round=None,
        neurotoxin_mask=None,
    ):
        # 1. Snapshot initial state (same as before)
        initial_client_params = parameters_to_vector(
            [global_client_model.state_dict()[name] for name in global_client_model.state_dict()]
        ).detach()
        
        server_model = main_server.server_models[self.id]
        server_opt = main_server.server_opts[self.id]
        server_before_sd = copy.deepcopy(server_model.state_dict())

        # 2. Timing/Poisoning check (Handles BadNet/DBA via data loading)
        if self.id < self.args.num_corrupt:
            self.check_poison_timing(round)

        if self.id < self.args.num_corrupt and self.args.attack == "A3FL":
            # 1. Initialize A3FL Search
            # We assume the agent can replicate the server architecture (Shadow Server)
            shadow_server = copy.deepcopy(main_server.server_models[self.id])
            a3fl_engine = A3FLTriggerSearch(self.args, global_client_model, shadow_server)
            
            # 2. Update trigger based on current model state
            self.optimized_trigger = a3fl_engine.optimize(self.train_loader, self.args.target_class)
            
            # 3. Apply this trigger during the rest of the round
        # You would modify your dataloader or a 'poison_batch' helper to use self.optimized_trigger

        global_client_model.train()
        # ... [Your existing optimizer setup] ...

        for local_epoch in range(self.args.local_ep):
            for i, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad(set_to_none=True)
                if self.id < self.args.num_corrupt and self.args.attack == "A3FL":
                    inputs = torch.clamp(inputs + self.optimized_trigger, 0, 1)
                    labels = torch.full_like(labels, self.args.target_class)
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)

                # Client Forward -> Server Forward/Backward -> Client Backward
                fx = global_client_model(inputs)
                smashed = fx.detach().clone().requires_grad_(True)
                dfx, loss_s, acc_s = main_server.train_on_smashed(self.id, smashed, labels)
                fx.backward(dfx)
                
                # ... [Existing Neurotoxin gradient masking logic] ...
                
                optimizer.step()

        # 3. Post-Training Attack Logic (PGD and Model Replacement)
        with torch.no_grad():
            after_train = parameters_to_vector(
                [global_client_model.state_dict()[name] for name in global_client_model.state_dict()]
            ).detach()
            
            # --- MODEL REPLACEMENT (Scaling) ---
            # Default scale is 1.0 (Baseline)
            scale = 1.0
            if self.id < self.args.num_corrupt and self.args.attack_mode == "replacement":
                # scale = Total Agents / (Agents per round * lr)
                # For simplicity in many SFL papers, we use N / m
                scale = self.args.num_agents / (self.args.num_agents * self.args.agent_frac)
            
            # Apply scaling to the delta
            client_update_vec = (after_train - initial_client_params) * scale

            # --- PGD CONSTRAINT ---
            if self.id < self.args.num_corrupt and self.args.attack_constraint == "pgd":
                # Calculate norm of the (potentially scaled) update
                eps = torch.norm(initial_client_params) * self.args.pgd_eps
                actual_norm = torch.norm(client_update_vec)
                
                if actual_norm > eps:
                    client_update_vec = (client_update_vec / actual_norm) * eps
            
            self.update = client_update_vec

            # --- Server Part (Scale server update as well if malicious) ---
            server_after_sd = {k: v.detach().clone() for k, v in server_model.state_dict().items()}
            server_update_dict = {}
            for k in server_before_sd.keys():
                delta = (server_after_sd[k] - server_before_sd[k].detach()) * scale
                server_update_dict[k] = delta.to(self.args.device)

            # Restore server state
            server_model.load_state_dict(server_before_sd)

        return self.update, server_update_dict

"""
Combination,Command Flags
BadNet + Baseline,--attack badnet --attack_mode baseline
BadNet + Replacement,--attack badnet --attack_mode replacement
DBA + PGD,--attack DBA --attack_mode baseline --attack_constraint pgd
DBA + Replacement + PGD,--attack DBA --attack_mode replacement --attack_constraint pgd
Component,Responsibility
Trigger (--attack),Handled by utils.poison_dataset. Decides what the model learns (BadNet static or DBA pieces).
Scaling (--attack_mode),Multiplies the delta. Decides how much impact the update has on the average.
Constraint (--attack_constraint),Uses PGD to clip the delta. Decides how stealthy the update remains.
"""


import torch
import torch.nn as nn
import copy

class A3FLTriggerSearch:
    def __init__(self, args, client_model, shadow_server_model):
        self.args = args
        self.device = args.device
        self.client_model = client_model
        self.shadow_server = shadow_server_model
        
        # Trigger Image initialization (e.g., 3x3 at bottom-right)
        self.trigger = torch.zeros((1, *args.input_shape)).to(self.device)
        self.trigger.requires_grad = True
        
        # Hyperparameters (from A3FL paper)
        self.lr = 0.01 
        self.inner_epochs = 5 # dm_adv_epochs
        self.noise_lambda = 0.01

    def apply_trigger(self, x):
        # Additive trigger: x_poison = clamp(x + trigger)
        return torch.clamp(x + self.trigger, 0, 1)

    def get_adv_shadow(self, dataloader):
        """ Step 1: Create the 'Unlearning' Shadow Server """
        adv_server = copy.deepcopy(self.shadow_server).to(self.device)
        adv_server.train()
        optimizer = torch.optim.SGD(adv_server.parameters(), lr=0.01)
        
        # Freeze client model during shadow server unlearning simulation
        self.client_model.eval()
        
        for _ in range(self.inner_epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                with torch.no_grad():
                    smashed = self.client_model(self.apply_trigger(inputs))
                
                # adv_server tries to predict the CORRECT label for poisoned inputs (Unlearning)
                outputs = adv_server(smashed)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return adv_server

    def optimize(self, dataloader, target_class):
        """ Step 2: Optimize Trigger Image """
        adv_server = self.get_adv_shadow(dataloader)
        self.client_model.eval()
        adv_server.eval()
        self.shadow_server.eval()

        for _ in range(10): # Outer optimization epochs
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                target = torch.full((inputs.size(0),), target_class, dtype=torch.long).to(self.device)
                
                # Forward with trigger
                poison_x = self.apply_trigger(inputs)
                smashed = self.client_model(poison_x)
                
                # A3FL Dual Loss: 
                # 1. Backdoor success on current shadow
                # 2. Backdoor success on 'Unlearning' shadow (Robustness)
                loss_current = nn.CrossEntropyLoss()(self.shadow_server(smashed), target)
                loss_robust = nn.CrossEntropyLoss()(adv_server(smashed), target)
                
                total_loss = loss_current + self.noise_lambda * loss_robust
                
                if self.trigger.grad is not None:
                    self.trigger.grad.zero_()
                total_loss.backward()
                
                # Sign-based update for the trigger image
                self.trigger.data -= self.lr * self.trigger.grad.sign()
                self.trigger.data.clamp_(0, 1)
        
        return self.trigger.detach()