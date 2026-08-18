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
        if self.args.data != "gtsrb":
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
       
        # ==========================================================
        # PHASE 2: LOCAL WEIGHT TRAINING (Standard Batch Loop)
        # ==========================================================
        for local_epoch in range(self.args.local_ep):
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()

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
