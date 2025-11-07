import copy
import time

import torch
import utils
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader

# class Agent():
#     def __init__(self, id, args, train_dataset=None, data_idxs=None, mask=None, backdoor_train_dataset=None,malicious_client_ids=None):
#         self.id = id
#         self.args = args
#         self.error = 0
#         self.hessian_metrix = []
#         # get datasets, fedemnist is handled differently as it doesn't come with pytorch
#         if self.args.data != "tinyimagenet":
#             self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)

#             # for backdoor attack, agent poisons his local dataset
#             if self.id < args.num_corrupt and self.args.attack != 'non' and self.args.data != 'sen140':

#                 self.clean_backup_dataset = copy.deepcopy(train_dataset)
#                 self.data_idxs = data_idxs
#                 if self.id in malicious_client_ids and self.args.attack != 'non':
#                     # Get the 0, 1, 2, 3 rank
#                     mal_rank = malicious_client_ids.index(self.id)
                    
#                     self.clean_backup_dataset = copy.deepcopy(train_dataset)
#                     self.data_idxs = data_idxs
                    
#                     # Poison the dataset with the specific DBA piece
#                     utils.poison_dataset(train_dataset, args, data_idxs, agent_idx=mal_rank)
#             elif self.id < args.num_corrupt and self.args.attack != 'non' and self.args.data == 'sen140':
#                 self.clean_backup_dataset = copy.deepcopy(train_dataset)
#                 self.data_idxs = data_idxs
#                 benign_part = data_idxs[:int(len(data_idxs) * (1 - self.args.poison_frac))]
#                 malicious_part = data_idxs[int(len(data_idxs) * (1 - self.args.poison_frac)):]

#                 self.train_dataset = utils.DatasetSplit_new(train_dataset, backdoor_train_dataset, benign_part, malicious_part, data_idxs)
#         else:
#             self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs, runtime_poison=True, args=args,
#                                                         client_id=id)
#         # get dataloader
#         self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
#                                        num_workers=args.num_workers, pin_memory=False, drop_last=True)
#         # size of local dataset
#         self.n_data = len(self.train_dataset)

#     def check_poison_timing(self, round):
#         if round > self.args.cease_poison:
#             self.train_dataset = utils.DatasetSplit(self.clean_backup_dataset, self.data_idxs)
#             self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
#                                            num_workers=self.args.num_workers, pin_memory=False, drop_last=True)

#     def local_train(self, global_model, criterion, round=None, neurotoxin_mask=None):
#         # print(len(self.train_dataset))
#         """ Do a local training over the received global model, return the update """
#         # start = time.time()
#         initial_global_model_params = parameters_to_vector(
#             [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
#         if self.id < self.args.num_corrupt:
#             self.check_poison_timing(round)
#         global_model.train()
#         optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr * (self.args.lr_decay) ** round,
#                                     weight_decay=self.args.wd, momentum=self.args.momentum)

#         regular_loss = 0.0
#         for local_epoch in range(self.args.local_ep):
#             start = time.time()
#             old_gradient = {}
#             old_gradient_mine = {}
#             old_params = {}
#             for i, (inputs, labels) in enumerate(self.train_loader):
#                 # if i == 0 and self.is_malicious:
#                 #     save_image(torch.cat([inputs[labels == self.args.target_class][:10]]), '%s_image.png' % self.id, normalize=True, nrow=10)
#                 optimizer.zero_grad()
#                 inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
#                                  labels.to(device=self.args.device, non_blocking=True)
#                 outputs = global_model(inputs)
#                 # outputs = outputs[:, :]
#                 minibatch_loss = criterion(outputs, labels)
#                 # print(minibatch_loss)
#                 minibatch_loss.backward()
#                 if self.args.attack == "neurotoxin" and len(neurotoxin_mask) and self.id < self.args.num_corrupt:
#                     for name, param in global_model.named_parameters():
#                         param.grad.data = neurotoxin_mask[name].to(self.args.device) * param.grad.data
#                 if self.args.attack == "r_neurotoxin" and len(neurotoxin_mask) and self.id < self.args.num_corrupt:
#                     for name, param in global_model.named_parameters():
#                         param.grad.data = (torch.ones_like(neurotoxin_mask[name].to(self.args.device))-neurotoxin_mask[name].to(self.args.device) ) * param.grad.data
#                 optimizer.step()

#                 if self.args.attack == 'pgd' and self.id < self.args.num_corrupt and (i == len(self.train_loader) - 1):
#                     if self.args.data == 'cifar10':
#                         eps = torch.norm(initial_global_model_params) * 0.1
#                     else:
#                         eps = torch.norm(initial_global_model_params)

#                     current_local_model_params = parameters_to_vector([global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
#                     norm_diff = torch.norm(current_local_model_params - initial_global_model_params)
#                     print('clip before: ', norm_diff)
#                     if norm_diff > eps:
#                         w_proj_vec = eps * (current_local_model_params - initial_global_model_params) / norm_diff + initial_global_model_params

#                         print('clip after: ', torch.norm(w_proj_vec - initial_global_model_params))

#                         new_state_dict = utils.vector_to_model_wo_load(w_proj_vec, global_model)    
#                         global_model.load_state_dict(new_state_dict)

#             end = time.time()
#             train_time = end - start
#             print("local epoch %d \t client: %d \t mal: %s \t loss: %.8f \t time: %.2f" % (local_epoch, self.id, str(self.is_malicious),
#                                                                      minibatch_loss, train_time))

#         with torch.no_grad():
#             after_train = parameters_to_vector(
#                 [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
#             self.update = after_train - initial_global_model_params

#             return self.update
#     def sfl_local_train(self, client_model, server_model, criterion, round=None):
#         """ 
#         Split Federated Learning version of local training.
#         The agent only trains the client-side layers.
#         """
#         if self.id < self.args.num_corrupt:
#             self.check_poison_timing(round)
#         initial_server_params = copy.deepcopy(server_model.state_dict())
#         client_model.train()
#         # The optimizer only manages the client-side model parameters
#         optimizer = torch.optim.SGD(client_model.parameters(), 
#                                     lr=self.args.client_lr * (self.args.lr_decay) ** round,
#                                     weight_decay=self.args.wd, 
#                                     momentum=self.args.momentum)

#         initial_client_params = parameters_to_vector(
#             [client_model.state_dict()[name] for name in client_model.state_dict()]).detach()
#         # To ensure ASR grows, the server optimizer must be consistent
#         server_optimizer = torch.optim.SGD(server_model.parameters(), 
#                                    lr=self.args.server_lr) 
                                   
#         for local_epoch in range(self.args.local_ep):
#             for i, (inputs, labels) in enumerate(self.train_loader):
#                 optimizer.zero_grad()
#                 server_optimizer.zero_grad() # ADDED: Clear server gradients
                
#                 # 1. Client Forward Pass
#                 inputs = inputs.to(device=self.args.device, non_blocking=True)
#                 labels = labels.to(device=self.args.device, non_blocking=True)
#                 fx = client_model(inputs)
                
#                 # 2. Communication
#                 smashed_data = fx.detach().requires_grad_(True)
                
#                 # 3. Server Forward Pass
#                 outputs = server_model(smashed_data)
#                 loss = criterion(outputs, labels)
                
#                 # 4. Server Backward Pass
#                 loss.backward()
                
#                 # 5. Communication: Get Gradients for Client
#                 grad_from_server = smashed_data.grad.clone().detach()
                
#                 # 6. UPDATE BOTH MODELS
#                 # First, the server updates its weights
#                 server_optimizer.step() # ADDED: This is what makes the server learn!
                
#                 # Second, the client updates its weights using the server's gradient
#                 fx.backward(grad_from_server)
#                 optimizer.step()
#         server_update = {}
#         for name, param in server_model.named_parameters():
#             server_update[name] = param.data - initial_server_params[name].data
#         with torch.no_grad():
#             after_train = parameters_to_vector(
#                 [client_model.state_dict()[name] for name in client_model.state_dict()]).detach()
#             # Return only the client-side update
#             self.update = after_train - initial_client_params
#             return self.update, server_update
#     def sfl_evaluate(self, client_model, server_model, criterion):
#         client_model.eval()
#         server_model.eval()
#         total_loss, correct, total = 0, 0, 0
        
#         with torch.no_grad():
#             for inputs, labels in self.train_loader: # Or self.test_loader
#                 inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                
#                 # Full SFL forward pass
#                 fx = client_model(inputs)
#                 outputs = server_model(fx)
                
#                 loss = criterion(outputs, labels)
#                 total_loss += loss.item()
                
#                 _, predicted = outputs.max(1)
#                 total += labels.size(0)
#                 correct += predicted.eq(labels).sum().item()
                
#         return total_loss / len(self.train_loader), 100. * correct / total


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

    def check_poison_timing(self, round):
        if round > self.args.cease_poison:
            self.train_dataset = utils.DatasetSplit(self.clean_backup_dataset, self.data_idxs)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                           num_workers=self.args.num_workers, pin_memory=False, drop_last=True)

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
                # # print(minibatch_loss)
                # minibatch_loss.backward()


                # base_loss = criterion(outputs, labels)
                
                # 2. Add AlignIns-Bypass Regularization (The key addition)
                if self.args.attack == 'alignins_bypass' and self.id < self.args.num_corrupt:
                    current_params = parameters_to_vector([p for p in global_model.parameters()])
                    current_update = current_params - initial_global_model_params
                    
                    # TDA Constraint: Keep cosine similarity high with global model
                    # Cosine(A, B) = (A dot B) / (||A|| * ||B||)
                    tda_loss = 1.0 - torch.nn.functional.cosine_similarity(
                        current_update.unsqueeze(0), 
                        initial_global_model_params.unsqueeze(0)
                    )
                    
                    # MPSA Constraint: Sign alignment
                    # We penalize updates that have different signs than the global model
                    # (Simplified proxy for Principal Sign p)
                    mpsa_loss = -torch.mean(current_update * torch.sign(initial_global_model_params))
                    
                    # Combine losses: Lambda values control how much you "sacrifice" 
                    # backdoor strength for stealth.
                    # loss = minibatch_loss + (0.5 * tda_loss) + (0.1 * mpsa_loss)
                    loss = minibatch_loss + (0.1 * tda_loss) + (0.01 * mpsa_loss)
                else:
                    loss = minibatch_loss

                loss.backward()
                # optimizer.step()

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
            raw_update = after_train - initial_global_model_params
            
            if self.args.attack == 'alignins_bypass' and self.id < self.args.num_corrupt:
                # Force the Top-K signs to match initial_global_model_params
                # This ensures the MPSA check passes almost perfectly
                k = int(len(raw_update) * 0.1) # Assuming k is 10% as per paper commonality
                topk_values, topk_indices = torch.topk(torch.abs(raw_update), k)
                
                # Principal Sign Proxy: Use the global model's sign as our target
                target_signs = torch.sign(initial_global_model_params[topk_indices])
                
                # Apply sign correction to the top-k most influential parameters
                raw_update[topk_indices] = torch.abs(raw_update[topk_indices]) * target_signs
            
            self.update = raw_update


            return self.update