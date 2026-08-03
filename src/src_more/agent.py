import copy
import time

import torch
import utils
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader

class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None, mask=None, backdoor_train_dataset=None):
        self.id = id
        self.args = args
        self.error = 0
        self.hessian_metrix = []

        
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


#creating trigger in the initialization phase
        self.is_malicious = self.id < args.num_corrupt
        
        if self.is_malicious:
            # Initialize A3FL specific variables
            img_shape = (3, 32, 32) if args.data == 'cifar10' else (1, 28, 28)
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
        grad_norm_sq = 0.0
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            self._raise_if_nonfinite(f"grad[{name}]", param.grad, round_idx, local_epoch, batch_idx)
            grad_norm_sq += param.grad.detach().float().pow(2).sum().item()
            grad_count += param.grad.numel()
        if grad_count == 0:
            return
        grad_norm = grad_norm_sq ** 0.5
        if not torch.isfinite(torch.tensor(grad_norm)):
            raise RuntimeError(
                f"Non-finite gradient norm detected "
                f"(client={self.id}, round={round_idx}, epoch={local_epoch}, batch={batch_idx})"
            )


    def get_adv_models(self, global_model):
        """ Simulates 'unlearning' to create adversarial shadow models (theta') """
        adv_models = []
        adv_ws = []
        cos_sim = nn.CosineSimilarity(dim=0)
        
        # We create multiple shadow models as per the Attacker class logic
        for _ in range(self.args.dm_adv_model_count):
            adv_model = copy.deepcopy(global_model).to(self.args.device)
            adv_model.train()
            adv_opt = torch.optim.SGD(adv_model.parameters(), lr=0.01, momentum=0.9)
            
            # Simulate unlearning for a few epochs
            for _ in range(self.args.dm_adv_epochs):
                for inputs, labels in self.train_loader:
                    inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                    # Poison the unlearning batch
                    poisoned = self.trigger.detach() * self.mask + (1 - self.mask) * inputs
                    # Defender target: classify as CLEAN labels
                    with torch.no_grad():
                        clean_labels = torch.argmax(global_model(inputs), dim=1)
                    
                    adv_opt.zero_grad()
                    nn.CrossEntropyLoss()(adv_model(poisoned), clean_labels).backward()
                    adv_opt.step()

            # Calculate the persistence weight (adv_w) using gradient similarity
            # This followsSnippet 1's logic of weighting shadow models
            sim_sum, count = 0.0, 0
            for name, param in adv_model.named_parameters():
                if 'conv' in name and param.grad is not None:
                    orig_param = dict(global_model.named_parameters())[name]
                    sim_sum += cos_sim(param.grad.view(-1), orig_param.grad.view(-1) if orig_param.grad is not None else torch.zeros_like(param.grad.view(-1)))
                    count += 1
            
            adv_models.append(adv_model)
            adv_ws.append(sim_sum / count if count > 0 else torch.tensor(1.0))
            
        return adv_models, adv_ws
    def check_poison_timing(self, round):
        if round > self.args.cease_poison:
            self.train_dataset = utils.DatasetSplit(self.clean_backup_dataset, self.data_idxs)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                           num_workers=self.args.num_workers, pin_memory=False, drop_last=True)

    def local_train(self, global_model, criterion, round=None, neurotoxin_mask=None):
        # print(len(self.train_dataset))
        """ Do a local training over the received global model, return the update """
        # start = time.time()
        initial_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.id < self.args.num_corrupt:
            self.check_poison_timing(round)
        global_model.train()
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr * (self.args.lr_decay) ** round,
                                    weight_decay=self.args.wd, momentum=self.args.momentum)

        


       
        if self.id < self.args.num_corrupt and self.args.attack == 'a3fl_opt':
                    # 1. Grab a single batch to serve as the "optimization proxy"
                    opt_inputs, _ = next(iter(self.train_loader))
                    opt_inputs = opt_inputs.to(self.args.device)
                    target_labels_opt = torch.full((opt_inputs.size(0),), self.args.target_class, device=self.args.device)

                    # 2. Generate Shadow Models (The A3FL "Armor")
                    adv_models, adv_ws = self.get_adv_models(global_model)
                    
                    # 3. Outer Search Loop
                    for _ in range(self.args.trigger_outter_epochs):
                        self.trigger = self.trigger.detach().requires_grad_(True)
                        p_input = self.trigger * self.mask + (1 - self.mask) * opt_inputs
                        
                        with torch.no_grad():
                            out_g = global_model(p_input)
                        
                        # Aggregate loss from all shadow models
                        loss_shadow = 0
                        for am, aw in zip(adv_models, adv_ws):
                            loss_shadow += aw * criterion(am(p_input), target_labels_opt)
                        
                        # Combine current model performance with shadow model persistence
                        t_loss = criterion(out_g.detach(), target_labels_opt) + \
                                (self.args.lambda_a3fl / len(adv_models)) * loss_shadow
                        t_loss.backward()
                        
                        with torch.no_grad():
                            if self.trigger.grad is not None:
                                self.trigger -= self.args.trigger_lr * self.trigger.grad.sign()
                                self.trigger.clamp_(0, 1)

                    # Cleanup memory immediately
                    del adv_models
                    torch.cuda.empty_cache()
                    



                # elif self.id < self.args.num_corrupt and self.args.attack == 'alignment':
                #     # 1. Prepare Shadow Model on CPU to save initial VRAM
                #     shadow_model = copy.deepcopy(global_model).to('cpu')
                #     shadow_model.train()
                #     shadow_optimizer = torch.optim.SGD(shadow_model.parameters(), lr=self.args.inner_lr)
                    
                #     # 2. Perform Unlearning on CPU
                #     inputs_cpu = inputs.to('cpu')
                #     with torch.no_grad():
                #         clean_labels_cpu = torch.argmax(shadow_model(inputs_cpu), dim=1)
                    
                #     shadow_optimizer.zero_grad()
                #     loss_unlearn = criterion(shadow_model(inputs_cpu), clean_labels_cpu)
                #     loss_unlearn.backward()
                #     shadow_optimizer.step()

                #     # --- 3. MOVE TO GPU FOR ALL CALCULATIONS ---
                #     shadow_model.to(self.args.device) 
                    
                #     # 4. Benign Reference
                #     outputs_benign = global_model(inputs)
                #     loss_benign = criterion(outputs_benign, clean_labels_cpu.to(self.args.device))
                #     loss_benign.backward(retain_graph=True)
                #     g_benign = torch.cat([p.grad.view(-1) for p in global_model.parameters()]).detach()

                    
                #     # --- 5. POISON OBJECTIVE (Memory-Optimized) ---
                #     target_labels = torch.full_like(labels, self.args.target_class).to(self.args.device)

                #     # Current Model (GPU)
                #     loss_p_current = criterion(global_model(inputs), target_labels)

                #     # Shadow Model Persistence (The "Armor" against 4 honest clients)
                #     shadow_model.to(self.args.device)
                #     with torch.no_grad(): 
                #         # We only need the OUTPUT of the shadow model to guide the global model
                #         # We do NOT need the shadow model's own gradient graph here
                #         outputs_shadow = shadow_model(inputs)

                #     # We optimize the GLOBAL model to produce the same result as the SHADOW model's target
                #     loss_p_shadow = criterion(outputs_shadow, target_labels)

                #     # Move shadow back to CPU immediately
                #     shadow_model.to('cpu') 

                #     # Combine losses for persistence
                #     loss_poison = loss_p_current + self.args.lambda_a3fl * loss_p_shadow

                #     # This creates the graph for global_model only, saving ~4GB VRAM
                #     loss_poison.backward(create_graph=True) 
                #     g_poison = torch.cat([p.grad.view(-1) for p in global_model.parameters()])

                #     # 6. Alignment Logic
                #     cos_sim = torch.nn.functional.cosine_similarity(g_poison, g_benign, dim=0)
                #     loss_dir = 1 - cos_sim
                #     mismatch = (g_poison * g_benign < 0).float()
                #     loss_sign = torch.sum(torch.abs(g_poison) * mismatch)

                #     total_loss = loss_poison + self.args.alpha * loss_dir + self.args.beta * loss_sign
                    
                #     # 7. Final Backward (Moves the actual global_model weights)
                #     optimizer.zero_grad()
                #     total_loss.backward()
                #     optimizer.step()
                    
                #     # --- 8. CLEANUP (Only move to CPU/Delete after ALL backwards are done) ---
                #     shadow_model.to('cpu') 
                #     del shadow_model, g_benign, g_poison, loss_p_shadow
                #     torch.cuda.empty_cache()
                    
                #     minibatch_loss = total_loss.item()



                #===================================
                
                #===================================


            #     if self.args.attack == "neurotoxin" and len(neurotoxin_mask) and self.id < self.args.num_corrupt:
            #         for name, param in global_model.named_parameters():
            #             param.grad.data = neurotoxin_mask[name].to(self.args.device) * param.grad.data
            #     if self.args.attack == "r_neurotoxin" and len(neurotoxin_mask) and self.id < self.args.num_corrupt:
            #         for name, param in global_model.named_parameters():
            #             param.grad.data = (torch.ones_like(neurotoxin_mask[name].to(self.args.device))-neurotoxin_mask[name].to(self.args.device) ) * param.grad.data
            #     optimizer.step()

            #     if self.args.attack == 'pgd' and self.id < self.args.num_corrupt and (i == len(self.train_loader) - 1):
            #         if self.args.data == 'cifar10':
            #             eps = torch.norm(initial_global_model_params) * 0.1
            #         else:
            #             eps = torch.norm(initial_global_model_params)

            #         current_local_model_params = parameters_to_vector([global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
            #         norm_diff = torch.norm(current_local_model_params - initial_global_model_params)
            #         print('clip before: ', norm_diff)
            #         if norm_diff > eps:
            #             w_proj_vec = eps * (current_local_model_params - initial_global_model_params) / norm_diff + initial_global_model_params

            #             print('clip after: ', torch.norm(w_proj_vec - initial_global_model_params))

            #             new_state_dict = utils.vector_to_model_wo_load(w_proj_vec, global_model)    
            #             global_model.load_state_dict(new_state_dict)

            # end = time.time()
            # train_time = end - start
            # print("local epoch %d \t client: %d \t mal: %s \t loss: %.8f \t time: %.2f" % (local_epoch, self.id, str(self.is_malicious),
            #                                                          minibatch_loss, train_time))




        if self.id < self.args.num_corrupt and self.args.attack == 'alignment':
            # Fetch a small proxy batch for optimization speed
            opt_inputs, _ = next(iter(self.train_loader))
            opt_inputs = opt_inputs[:16].to(self.args.device) # Subset for speed
            target_labels_opt = torch.full((opt_inputs.size(0),), self.args.target_class, device=self.args.device)

            # Get Benign Reference for Search Alignment
            global_model.zero_grad()
            with torch.no_grad():
                clean_preds_ref = torch.argmax(global_model(opt_inputs), dim=1)
            criterion(global_model(opt_inputs), clean_preds_ref).backward()
            g_benign_ref = torch.cat([p.grad.view(-1) for p in global_model.parameters()]).detach()
            global_model.zero_grad()

            # Generate Shadow Models (The Persistence "Armor")
            adv_models, adv_ws = self.get_adv_models(global_model)
            
            for _ in range(self.args.trigger_outter_epochs):
                self.trigger = self.trigger.detach().requires_grad_(True)
                p_input = self.trigger * self.mask + (1 - self.mask) * opt_inputs
                
                # --- Search Alignment Logic ---
                global_model.zero_grad()
                loss_p_search = criterion(global_model(p_input), target_labels_opt)
                # Create graph allows trigger pixels to "see" how they affect the gradient
                loss_p_search.backward(create_graph=True)
                g_poison_search = torch.cat([p.grad.view(-1) for p in global_model.parameters()])
                
                loss_dir_s = 1 - torch.nn.functional.cosine_similarity(g_poison_search, g_benign_ref, dim=0, eps=1e-8)
                loss_sign_s = torch.sum(torch.abs(g_poison_search) * (g_poison_search * g_benign_ref < 0).float())

                # --- Shadow Persistence Logic ---
                loss_shadow = 0
                for am, aw in zip(adv_models, adv_ws):
                    loss_shadow += aw * criterion(am(p_input), target_labels_opt)
                
                # Combined Search Formula: Accuracy + Persistence + Stealth
                total_t_loss = loss_p_search + \
                            (self.args.lambda_a3fl / len(adv_models)) * loss_shadow + \
                            self.args.alignment_alpha * loss_dir_s + \
                            self.args.alignment_beta * loss_sign_s

                global_model.zero_grad()
                total_t_loss.backward()
                
                with torch.no_grad():
                    if self.trigger.grad is not None:
                        self.trigger -= self.args.trigger_lr * self.trigger.grad.sign()
                        self.trigger.clamp_(0, 1)

            del adv_models, g_benign_ref, g_poison_search
            torch.cuda.empty_cache()
        # ==========================================================
        # PHASE 2: LOCAL WEIGHT TRAINING (Standard Batch Loop)
        # ==========================================================
        for local_epoch in range(self.args.local_ep):
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()

                if self.id < self.args.num_corrupt and self.args.attack == 'a3fl_opt':
                    # --- ALIGNED POISONING PATH ---
                    # 1. Benign Reference (Stealth)
                    with torch.no_grad():
                        clean_preds = torch.argmax(global_model(inputs), dim=1)
                    criterion(global_model(inputs), clean_preds).backward()
                    g_benign = torch.cat([p.grad.view(-1) for p in global_model.parameters()]).detach()
                    optimizer.zero_grad()

                    # 2. Poison Gradient (Aligned)
                    p_inputs = self.trigger.detach() * self.mask + (1 - self.mask) * inputs
                    target_labels = torch.full_like(labels, self.args.target_class)
                    
                    loss_p = criterion(global_model(p_inputs), target_labels)
                    loss_p.backward(create_graph=True) 
                    g_poison = torch.cat([p.grad.view(-1) for p in global_model.parameters()])

                    
                    
                    total_loss = loss_p 
                    optimizer.zero_grad()
                    total_loss.backward()
                    
                    # Apply Neurotoxin if masks exist
                    if neurotoxin_mask:
                        for name, param in global_model.named_parameters():
                            if name in neurotoxin_mask:
                                param.grad.data *= neurotoxin_mask[name].to(self.args.device)
                    
                    minibatch_loss = total_loss.item()
                    del g_benign, g_poison

                elif self.id < self.args.num_corrupt and self.args.attack == 'alignment':
                    # --- ALIGNED POISONING PATH ---
                    # 1. Benign Reference (Stealth)
                    with torch.no_grad():
                        clean_preds = torch.argmax(global_model(inputs), dim=1)
                    criterion(global_model(inputs), clean_preds).backward()
                    g_benign = torch.cat([p.grad.view(-1) for p in global_model.parameters()]).detach()
                    optimizer.zero_grad()

                    # 2. Poison Gradient (Aligned)
                    p_inputs = self.trigger.detach() * self.mask + (1 - self.mask) * inputs
                    target_labels = torch.full_like(labels, self.args.target_class)
                    
                    loss_p = criterion(global_model(p_inputs), target_labels)
                    loss_p.backward(create_graph=True) 
                    g_poison = torch.cat([p.grad.view(-1) for p in global_model.parameters()])
                    self._raise_if_nonfinite("alignment/g_benign", g_benign, round, local_epoch, i)
                    self._raise_if_nonfinite("alignment/loss_p", loss_p, round, local_epoch, i)
                    self._raise_if_nonfinite("alignment/g_poison", g_poison, round, local_epoch, i)

                    # 3. Alignment Logic
                    loss_dir = 1 - torch.nn.functional.cosine_similarity(g_poison, g_benign, dim=0, eps=1e-8)
                    loss_sign = torch.sum(torch.abs(g_poison) * (g_poison * g_benign < 0).float())
                    self._raise_if_nonfinite("alignment/loss_dir", loss_dir, round, local_epoch, i)
                    self._raise_if_nonfinite("alignment/loss_sign", loss_sign, round, local_epoch, i)
                    
                    total_loss = loss_p + self.args.alignment_alpha * loss_dir + self.args.alignment_beta * loss_sign
                    self._raise_if_nonfinite("alignment/total_loss", total_loss, round, local_epoch, i)
                    optimizer.zero_grad()
                    total_loss.backward()
                    self._raise_if_nonfinite_grads(global_model, round, local_epoch, i)
                    
                    # Apply Neurotoxin if masks exist
                    if neurotoxin_mask:
                        for name, param in global_model.named_parameters():
                            if name in neurotoxin_mask:
                                param.grad.data *= neurotoxin_mask[name].to(self.args.device)
                    self._raise_if_nonfinite_grads(global_model, round, local_epoch, i)
                    
                    minibatch_loss = total_loss.item()
                    del g_benign, g_poison
                
                else:
                    # --- HONEST PATH ---
                    outputs = global_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    minibatch_loss = loss.item()

                optimizer.step()
                
                if i % 20 == 0:
                    torch.cuda.empty_cache()

            print(f"Epoch {local_epoch} done. Loss: {minibatch_loss:.4f}")

        # Return the local model update (after - before)
        with torch.no_grad():
            after_train = parameters_to_vector([global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
            self.update = after_train - initial_params
        return self.update
