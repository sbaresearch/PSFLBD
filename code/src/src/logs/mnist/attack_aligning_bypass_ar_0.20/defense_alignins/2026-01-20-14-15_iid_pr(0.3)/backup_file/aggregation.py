import copy

import torch
from torch.nn.utils import parameters_to_vector
import numpy as np
import logging
from utils import vector_to_model, vector_to_name_param

import sklearn.metrics.pairwise as smp
from geom_median.torch import compute_geometric_median 


import torch
from typing import Dict, List, Tuple

TensorDict = Dict[str, torch.Tensor]


class Aggregation():
    def __init__(self, agent_data_sizes, n_params, args):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.server_lr = args.server_lr
        self.n_params = n_params
        self.total_data = sum([agent_data_sizes[i] for i in agent_data_sizes])
        
        if self.args.aggr == 'foolsgold':
            self.memory_dict = dict()
            self.wv_history = []

    
        
    # def aggregate_server_updates(self, global_server_model, agent_server_updates):
    #     """ Perform Parallel SFL server-side aggregation. """
        
    #     # 1. Calculate total data for ONLY the participating agents this round
    #     # This is where 'agent_server_updates' lives!
    #     round_total_data = sum([self.agent_data_sizes[idx] for idx in agent_server_updates.keys()])
        
    #     # 2. Initialize an accumulator (grad_sum) for the weights
    #     grad_sum = {}
    #     first_agent_id = list(agent_server_updates.keys())[0]
    #     for name in agent_server_updates[first_agent_id]:
    #         grad_sum[name] = torch.zeros_like(agent_server_updates[first_agent_id][name])

    #     # 3. Apply FedAvg: update * (local_n / round_total_n)
    #     for agent_id, update in agent_server_updates.items():
    #         weight = self.agent_data_sizes[agent_id] / round_total_data
    #         for name in grad_sum:
    #             grad_sum[name] += update[name].to(self.args.device) * weight

    #     # 4. Apply the aggregated delta to the global server model
    #     new_state_dict = global_server_model.state_dict()
    #     for name in grad_sum:
    #         new_state_dict[name] += grad_sum[name]
        
    #     global_server_model.load_state_dict(new_state_dict)
    #     return global_server_model

    def aggregate_server_updates(self, global_server_model, agent_server_updates):
        """
        MainServer aggregation with optional defenses (same menu as FedServer).
        Keeps FedAvg weighting by pre-scaling each agent update by its data fraction.
        """

        # 0) Basic checks
        if len(agent_server_updates) == 0:
            return global_server_model

        # 1) Fixed param order
        param_order = get_param_order_from_model(global_server_model)


        # 3) Convert each agent's server update (dict) -> vector, and pre-weight it
        agent_update_vecs = {}
        for agent_id, upd_dict in agent_server_updates.items():
            w = self.agent_data_sizes[agent_id] / round_total_data  # FedAvg weight
            v = update_dict_to_vector(upd_dict, param_order, device=self.args.device)
            agent_update_vecs[agent_id] = v * float(w)  # pre-scale to preserve FedAvg weighting

        # 4) Run aggregation/defense (same switch style as FedServer)
        # NOTE: after pre-weighting, simple average == weighted FedAvg
        if self.args.aggr_server in ("avg", "rlr", "lockdown"):
            aggregated_vec = self.agg_avg(agent_update_vecs)

        elif self.args.aggr_server == "alignins":
            cur_global_params = torch.nn.utils.parameters_to_vector(
                [global_server_model.state_dict()[n] for n in param_order]
            ).detach().to(self.args.device)
            aggregated_vec = self.agg_alignins(agent_update_vecs, cur_global_params)

        elif self.args.aggr_server == "mmetric":
            cur_global_params = torch.nn.utils.parameters_to_vector(
                [global_server_model.state_dict()[n] for n in param_order]
            ).detach().to(self.args.device)
            aggregated_vec = self.agg_mul_metric(agent_update_vecs, global_server_model, cur_global_params)

        elif self.args.aggr_server == "foolsgold":
            aggregated_vec = self.agg_foolsgold(agent_update_vecs)

        elif self.args.aggr_server == "signguard":
            aggregated_vec = self.agg_signguard(agent_update_vecs)

        elif self.args.aggr_server == "mkrum":
            aggregated_vec = self.agg_mkrum(agent_update_vecs)

        elif self.args.aggr_server == "rfa":
            aggregated_vec = self.agg_rfa(agent_update_vecs)

        else:
            # Fallback: behave like your original FedAvg
            aggregated_vec = self.agg_avg(agent_update_vecs)

        # 5) Apply aggregated delta to the global server model (dict form)
        new_state_dict = copy.deepcopy(global_server_model.state_dict())
        delta_dict = vector_to_update_dict(aggregated_vec, new_state_dict, param_order)

        for name in param_order:
            new_state_dict[name] = new_state_dict[name] + delta_dict[name].to(new_state_dict[name].device)

        global_server_model.load_state_dict(new_state_dict)
        return global_server_model

    def aggregate_updates(self, global_model, agent_updates_dict):


        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(self.args.device)
        if self.args.aggr != "rlr":
            lr_vector = lr_vector
        else:
            lr_vector, _ = self.compute_robustLR(agent_updates_dict)
        # mask = torch.ones_like(agent_updates_dict[0])
        aggregated_updates = 0
        cur_global_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.args.aggr=='avg' or self.args.aggr == 'rlr' or self.args.aggr == 'lockdown':          
            aggregated_updates = self.agg_avg(agent_updates_dict)

        elif self.args.aggr == 'alignins':
            aggregated_updates = self.agg_alignins(agent_updates_dict, cur_global_params)
        elif self.args.aggr == 'mmetric':
            aggregated_updates = self.agg_mul_metric(agent_updates_dict, global_model, cur_global_params)
        elif self.args.aggr == 'foolsgold':
            aggregated_updates = self.agg_foolsgold(agent_updates_dict)
        elif self.args.aggr == 'signguard':
            aggregated_updates = self.agg_signguard(agent_updates_dict)
        elif self.args.aggr == "mkrum":
            aggregated_updates = self.agg_mkrum(agent_updates_dict)
        elif self.args.aggr == "rfa":
            aggregated_updates = self.agg_rfa(agent_updates_dict)
        neurotoxin_mask = {}
        updates_dict = vector_to_name_param(aggregated_updates, copy.deepcopy(global_model.state_dict()))
        for name in updates_dict:
            updates = updates_dict[name].abs().view(-1)
            gradients_length = torch.numel(updates)
            _, indices = torch.topk(-1 * updates, int(gradients_length * self.args.dense_ratio))
            mask_flat = torch.zeros(gradients_length)
            mask_flat[indices.cpu()] = 1
            neurotoxin_mask[name] = (mask_flat.reshape(updates_dict[name].size()))

        cur_global_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float()
        vector_to_model(new_global_params, global_model)
        return updates_dict, neurotoxin_mask

    def agg_rfa(self, agent_updates_dict):
        local_updates = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)

        n = len(local_updates)
        temp_updates = torch.stack(local_updates, dim=0)
        weights = torch.ones(n).to(self.args.device)  
        gw = compute_geometric_median(local_updates, weights).median
        for i in range(2):
            weights = torch.mul(weights, torch.exp(-1.0*torch.norm(temp_updates-gw, dim=1)))
            gw = compute_geometric_median(local_updates, weights).median

        aggregated_model = gw
        return aggregated_model

    def agg_alignins(self, agent_updates_dict, flat_global_model):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)

        tda_list = []
        mpsa_list = []
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
    
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())


        logging.info('TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])


        ######## MZ-score calculation ########
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)

        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)

        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
        
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)

        logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])

        ######## Anomaly detection with MZ score ########

        benign_idx1 = set([i for i in range(num_chosen_clients)])
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))


        # --- DEBUG LOGGING ---
        malicious_indices = set(range(len(malicious_id)))
        passed_mpsa = [i for i in benign_idx1 if i in malicious_indices]
        passed_tda = [i for i in benign_idx2 if i in malicious_indices]

        logging.info(f">>> Malicious Clients: {len(malicious_id)}")
        logging.info(f">>> Malicious IDs passing MPSA: {passed_mpsa}")
        logging.info(f">>> Malicious IDs passing TDA: {passed_tda}")

        benign_set = benign_idx2.intersection(benign_idx1)
        
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        ######## Post-filtering model clipping ########
        
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        benign_updates = torch.stack(local_updates, dim=0)
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        # del grad_norm
        
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        current_dict = {}
        for idx in benign_idx:
            current_dict[chosen_clients[idx]] = benign_updates[idx]

        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """

        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data
        return  sm_updates / total_data

    
    def agg_mkrum(self, agent_updates_dict):
        krum_param_m = 10
        def _compute_krum_score( vec_grad_list, byzantine_client_num):
            krum_scores = []
            num_client = len(vec_grad_list)
            for i in range(0, num_client):
                dists = []
                for j in range(0, num_client):
                    if i != j:
                        dists.append(
                            torch.norm(vec_grad_list[i]- vec_grad_list[j])
                            .item() ** 2
                        )
                dists.sort()  # ascending
                score = dists[0: num_client - byzantine_client_num - 2]
                krum_scores.append(sum(score))
            return krum_scores

        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            # local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        # Compute list of scores
        __nbworkers = len(agent_updates_dict)
        krum_scores = _compute_krum_score(agent_updates_dict, self.args.num_corrupt)
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0: krum_param_m]

        print('%d clients are selected' % len(score_index))
        return_updates = [agent_updates_dict[i] for i in score_index]


        return sum(return_updates)/len(return_updates)

    def compute_robustLR(self, agent_updates_dict):

        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        mask=torch.zeros_like(sm_of_signs)
        mask[sm_of_signs < self.args.theta] = 0
        mask[sm_of_signs >= self.args.theta] = 1
        sm_of_signs[sm_of_signs < self.args.theta] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.theta] = self.server_lr
        return sm_of_signs.to(self.args.device), mask

    def agg_mul_metric(self, agent_updates_dict, global_model, flat_global_model):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)

        vectorize_nets = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]

        cos_dis = [0.0] * len(vectorize_nets)
        length_dis = [0.0] * len(vectorize_nets)
        manhattan_dis = [0.0] * len(vectorize_nets)
        for i, g_i in enumerate(vectorize_nets):
            for j in range(len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]

                    cosine_distance = float(
                        (1 - np.dot(g_i, g_j) / (np.linalg.norm(g_i) * np.linalg.norm(g_j))) ** 2)   #Compute the different value of cosine distance
                    manhattan_distance = float(np.linalg.norm(g_i - g_j, ord=1))    #Compute the different value of Manhattan distance
                    length_distance = np.abs(float(np.linalg.norm(g_i) - np.linalg.norm(g_j)))    #Compute the different value of Euclidean distance

                    cos_dis[i] += cosine_distance
                    length_dis[i] += length_distance
                    manhattan_dis[i] += manhattan_distance

        tri_distance = np.vstack([cos_dis, manhattan_dis, length_dis]).T

        cov_matrix = np.cov(tri_distance.T)
        inv_matrix = np.linalg.inv(cov_matrix)

        ma_distances = []
        for i, g_i in enumerate(vectorize_nets):
            t = tri_distance[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)

        scores = ma_distances
        print(scores)

        p = 0.3
        p_num = p*len(scores)
        topk_ind = np.argpartition(scores, int(p_num))[:int(p_num)]   #sort

        print(topk_ind)
        current_dict = {}

        for idx in topk_ind:
            current_dict[chosen_clients[idx]] = agent_updates_dict[chosen_clients[idx]]

        update = self.agg_avg(current_dict)

        return update
   
    def agg_foolsgold(self, agent_updates_dict):

        def foolsgold(updates):
            """
            :param updates:
            :return: compute similatiry and return weightings
            """
            n_clients = updates.shape[0]
            cs = smp.cosine_similarity(updates) - np.eye(n_clients)

            maxcs = np.max(cs, axis=1)
            # pardoning
            for i in range(n_clients):
                for j in range(n_clients):
                    if i == j:
                        continue
                    if maxcs[i] < maxcs[j]:
                        cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
            wv = 1 - (np.max(cs, axis=1))

            wv[wv > 1] = 1
            wv[wv < 0] = 0

            alpha = np.max(cs, axis=1)

            # Rescale so that max value is wv
            wv = wv / np.max(wv)
            wv[(wv == 1)] = .99

            # Logit function
            wv = (np.log(wv / (1 - wv)) + 0.5)
            wv[(np.isinf(wv) + wv > 1)] = 1
            wv[(wv < 0)] = 0

            # wv is the weight
            return wv, alpha

        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        names = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)

        client_updates = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]
        update_len = np.array(client_updates[0].shape).prod()
        # print("client_updates size", client_models[0].parameters())
        # update_len = len(client_updates)
        # if self.memory is None:
        #     self.memory = np.zeros((self.num_clients, update_len))
        if len(names) < len(client_updates):
            names = np.append([-1], names)  # put in adv

        num_clients = num_chosen_clients
        memory = np.zeros((num_clients, update_len))
        updates = np.zeros((num_clients, update_len))

        for i in range(len(client_updates)):
            # updates[i] = np.reshape(client_updates[i][-2].cpu().data.numpy(), (update_len))
            updates[i] = np.reshape(client_updates[i], (update_len))
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]] += updates[i]
            else:
                self.memory_dict[names[i]] = copy.deepcopy(updates[i])
            memory[i] = self.memory_dict[names[i]]
        # self.memory += updates
        use_memory = False

        if use_memory:
            wv, alpha = foolsgold(None)  # Use FG
        else:
            wv, alpha = foolsgold(updates)  # Use FG
        # logger.info(f'[foolsgold agg] wv: {wv}')
        self.wv_history.append(wv)

        print(len(client_updates), len(wv))


        weighted_updates = [update * wv[i] for update, i in zip(agent_updates_dict.values(), range(len(wv)))]

        aggregated_model = torch.mean(torch.stack(weighted_updates, dim=0), dim=0)

        print(aggregated_model.shape)

        return aggregated_model

def get_param_order_from_model(model) -> List[str]:
        """
        Fixed param order for vectorization. Use this ONCE and reuse it everywhere.
        """
        return list(model.state_dict().keys())


def state_dict_like_to_vector(
    update_dict: TensorDict,
    param_order: List[str],
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Flatten a dict of tensors (e.g., server-side weight deltas) into one 1D vector
    using the given param_order.

    Assumes update_dict contains exactly the keys in param_order (or a superset).
    """
    vec_parts = []
    for name in param_order:
        if name not in update_dict:
            raise KeyError(f"Missing parameter '{name}' in update_dict.")
        t = update_dict[name]
        if device is not None:
            t = t.to(device)
        if dtype is not None:
            t = t.to(dtype=dtype)
        vec_parts.append(t.contiguous().view(-1))
    return torch.cat(vec_parts, dim=0)


def vector_to_state_dict_like(
    vec: torch.Tensor,
    template_state_dict: TensorDict,
    param_order: List[str],
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> TensorDict:
    """
    Convert a flat vector back into a dict of tensors with shapes matching
    template_state_dict, using the given param_order.
    """
    out: TensorDict = {}
    offset = 0

    if device is not None:
        vec = vec.to(device)
    if dtype is not None:
        vec = vec.to(dtype=dtype)

    for name in param_order:
        if name not in template_state_dict:
            raise KeyError(f"Parameter '{name}' not found in template_state_dict.")

        ref = template_state_dict[name]
        numel = ref.numel()

        if offset + numel > vec.numel():
            raise ValueError(
                f"Vector too short. Need at least {offset+numel} elements, got {vec.numel()}."
            )

        chunk = vec[offset : offset + numel]
        # Match dtype/device of template unless explicitly overridden
        chunk = chunk.view(ref.shape)
        if dtype is None:
            chunk = chunk.to(dtype=ref.dtype)
        if device is None:
            chunk = chunk.to(device=ref.device)

        out[name] = chunk
        offset += numel

    if offset != vec.numel():
        # This is often a bug (wrong param_order or wrong vector), so fail loudly.
        raise ValueError(
            f"Vector has extra elements. Used {offset}, but vec has {vec.numel()}."
        )

    return out


def apply_update_dict_inplace(model, update_dict: TensorDict):
    """
    Adds update_dict[name] to model.state_dict()[name] in-place (like your FedAvg code).
    Assumes update_dict tensors already on correct device.
    """
    sd = model.state_dict()
    for name, delta in update_dict.items():
        sd[name] = sd[name] + delta.to(sd[name].device)
    model.load_state_dict(sd)


def apply_update_vector_inplace(
    model,
    update_vec: torch.Tensor,
    param_order: List[str],
):
    """
    Adds a flat update vector to model parameters (in-place) using param_order.
    """
    sd = model.state_dict()
    delta_sd = vector_to_state_dict_like(update_vec, sd, param_order)
    apply_update_dict_inplace(model, delta_sd)