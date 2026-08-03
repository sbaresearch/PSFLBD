import inspect

import torch
import torch.nn as nn
from torchvision import models
import copy
import numpy as np
from aggregation import *
import torch.nn.functional as F
from torchvision.models.googlenet import BasicConv2d
from safesplit_defense import SafeSplitDefense
class SplitModel(nn.Module):
    """
    A thin wrapper that applies a list of modules in sequence.
    Uses ModuleList (not Sequential) to keep parameter names stable-ish and
    make debugging easier.
    """
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def get_resnet18_custom(data: str = "tinyimagenet", input_size: int = 32, num_classes: int = 200) -> nn.Module:
    """
    ResNet-18 modified for CIFAR-like inputs (3x32x32) and MNIST (1x28x28):
      - conv1: 3x3 stride 1
      - maxpool: Identity (common for small images)
      - fc: adjusted to num_classes
    """
    model = models.resnet18()

    in_channels = 1 if data in ["mnist", "fmnist", "femnist"] else 3

    # if input_size >= 224:
    #     # Keep the default 7x7 s2 conv1 and maxpool
    #     if in_channels == 1:
    #         # adapt conv1 to 1 channel by averaging RGB weights
    #         old_w = model.conv1.weight.data
    #         new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #         with torch.no_grad():
    #             if old_w.shape[1] == 3:
    #                 new_conv.weight[:] = old_w.mean(dim=1, keepdim=True)
    #             else:
    #                 nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
    #         model.conv1 = new_conv
    # # else: 3-channel ETS/TinyImageNet: no change
    # else:
    model.conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet18_split(
    data: str,
    num_classes: int,
    input_size:int = 32,
    split_point: str = "layer2",  # "layer1" (current), "layer2", "layer3"
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """
    Split ResNet-18 at a chosen stage so the client has more layers.

    split_point:
      - "layer1": client up to layer1 (your current split)
      - "layer2": client up to layer2 (more client compute)
      - "layer3": client up to layer3 (even more client compute)

    Returns (client_model, server_model, full_model).
    """
    full_model = get_resnet18_custom(data=data, input_size=input_size, num_classes=num_classes)

    stem = [
        full_model.conv1,
        full_model.bn1,
        full_model.relu,
        full_model.maxpool,
    ]

    if split_point == "layer1":
        client_layers = stem + [full_model.layer1]
        server_layers = [
            full_model.layer2,
            full_model.layer3,
            full_model.layer4,
            full_model.avgpool,
            nn.Flatten(),
            full_model.fc,
        ]

    elif split_point == "layer2":
        # client gets layer1 + layer2
        client_layers = stem + [full_model.layer1, full_model.layer2]
        server_layers = [
            full_model.layer3,
            full_model.layer4,
            full_model.avgpool,
            nn.Flatten(),
            full_model.fc,
        ]

    elif split_point == "layer3":
        # client gets layer1 + layer2 + layer3
        client_layers = stem + [full_model.layer1, full_model.layer2, full_model.layer3]
        server_layers = [
            full_model.layer4,
            full_model.avgpool,
            nn.Flatten(),
            full_model.fc,
        ]

    else:
        raise ValueError("split_point must be one of: 'layer1', 'layer2', 'layer3'")

    client_part = SplitModel(client_layers)
    server_part = SplitModel(server_layers)

    return client_part, server_part, full_model

def adapt_googlenet_for_small_inputs(model: nn.Module) -> nn.Module:
    """
    Adapt GoogLeNet to work well with 32x32 inputs:
    - Use a 3x3 conv, stride=1 instead of 7x7/stride=2
    - Remove the first maxpool to preserve spatial resolution
    """
    model.conv1 = BasicConv2d(3, 64, kernel_size=3, stride=1, padding=1)
    model.maxpool1 = nn.Identity()
    return model


def _create_googlenet(num_classes: int, aux_logits: bool):
    # Make the constructor compatible with both old (pretrained) and new (weights) APIs
    fn = models.googlenet
    sig = inspect.signature(fn)
    kwargs = {"num_classes": num_classes, "aux_logits": aux_logits}
    if "weights" in sig.parameters:
        kwargs["weights"] = None # new API
    elif "pretrained" in sig.parameters:
        kwargs["pretrained"] = False # old API
    return fn(**kwargs)

def replace_bn_with_gn(module, num_groups=8):
    """Recursively replace all BatchNorm2d with GroupNorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            # Ensure num_groups divides num_channels
            g = min(num_groups, num_channels)
            while num_channels % g != 0:
                g -= 1
            setattr(module, name, nn.GroupNorm(g, num_channels))
        else:
            replace_bn_with_gn(child, num_groups)
    return module

def get_googlenet_custom(data: str, num_classes: int, small_input: bool = True) -> nn.Module:
    m = _create_googlenet(num_classes=num_classes, aux_logits=False)
    if small_input:
        m = adapt_googlenet_for_small_inputs(m)
    m = replace_bn_with_gn(m)  # <-- add this
    return m


def get_googlenet_split(
    data: str,
    num_classes: int,
    split_point: str = "inception3", # "inception3", "inception4", "inception5"
    small_input: bool = True
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """
    Split GoogLeNet for split federated learning.
    - "inception3": client up to inception3b + maxpool3 (small client)
    - "inception4": client up to inception4d (medium client)
    - "inception5": client up to inception5a (larger client)
    Returns (client_model, server_model, full_model)
    """
    m = get_googlenet_custom(data=data, num_classes=num_classes, small_input=small_input)

    stem = [m.conv1, m.maxpool1, m.conv2, m.conv3, m.maxpool2]
    tail = [m.avgpool, nn.Flatten(), m.dropout, m.fc]

    if split_point == "inception3":
        client_layers = stem + [m.inception3a, m.inception3b, m.maxpool3]
        server_layers = [
            m.inception4a, m.inception4b, m.inception4c, m.inception4d, m.inception4e,
            m.maxpool4, m.inception5a, m.inception5b
        ] + tail

    elif split_point == "inception4":
        client_layers = stem + [
            m.inception3a, m.inception3b, m.maxpool3,
            m.inception4a, m.inception4b, m.inception4c, m.inception4d
        ]
        server_layers = [m.inception4e, m.maxpool4, m.inception5a, m.inception5b] + tail

    elif split_point == "inception5":
        client_layers = stem + [
            m.inception3a, m.inception3b, m.maxpool3,
            m.inception4a, m.inception4b, m.inception4c, m.inception4d, m.inception4e, m.maxpool4,
            m.inception5a
        ]
        server_layers = [m.inception5b] + tail

    else:
        raise ValueError("split_point must be one of: 'inception3', 'inception4', 'inception5'")

    client_model = SplitModel(client_layers)
    server_model = SplitModel(server_layers)
    return client_model, server_model, m

def get_resnet9_split(data: str, args) -> tuple[nn.Module, nn.Module, nn.Module]:
    """
    Same pattern: build ONE full model, then split.
    Adjust this split based on your ResNet9 definition.
    """
    from resnet import ResNet9

    in_channels = 1 if data in ["mnist", "fmnist", "femnist"] else 3
    num_classes = 10  # CIFAR10/MNIST

    full_model = ResNet9(in_channels, num_classes=num_classes, args=args)

    # Example split (matches your idea)
    client_layers = [full_model.prep, full_model.layer1]
    server_layers = [full_model.layer2, full_model.layer3, full_model.layer4, full_model.fc]

    client_part = SplitModel(client_layers)
    server_part = SplitModel(server_layers)

    return client_part, server_part, full_model


def get_split_models(data: str, args):
    """
    Single entry point:
      returns client_part, server_part, full_model
    so you DON'T accidentally create two different full models for client/server.
    """
    if data in ("cifar10", "mnist","fmnist"):
        # choose one:
        # return get_resnet9_split(data, args)
        if args.model == "resnet18":
            return get_resnet18_split(data=data, num_classes=10,split_point = args.split_point)
        else:
            return get_googlenet_split(data=data, num_classes=10)

    
    if data == "ets":
        return get_resnet18_split(data=data, num_classes=164, split_point=args.split_point)

    if data == "gtsrb":
        if args.model == "resnet18":
            return get_resnet18_split(data=data, num_classes=43, split_point=args.split_point)
        else:
            return get_googlenet_split(data=data, num_classes=43)

    if data == "cifar100":
        return get_resnet18_split(data=data, num_classes=100,split_point = args.split_point)

    # if data == "femnist":
    #     return get_resnet18_split(data=data, num_classes=62,split_point = args.split_point)

    if data == "tinyimagenet":
        return get_resnet18_split(data=data, num_classes=200,split_point = args.split_point)

    raise ValueError(f"No split model defined for data='{data}'")


def sanity_check_split(data: str, args, device: str = "cpu"):
    """
    Quick check: full_model(x) should match server(client(x)) in shape and run.
    (Values won't match unless weights are identical and both paths use same modules,
     which they do here.)
    """
    client, server, full = get_split_models(data, args)

    client = client.to(device)
    server = server.to(device)
    full = full.to(device)

    # Pick input size based on dataset
    if data in ("mnist","fmnist"):
        x = torch.randn(4, 1, 28, 28, device=device)
    elif data in ("cifar10", "cifar100", "gtsrb", "ets"):
        x = torch.randn(4, 3, 32, 32, device=device)
    elif data == "tinyimagenet":
        x = torch.randn(4, 3, 64, 64, device=device)
    else:
        x = torch.randn(4, 3, 32, 32, device=device)

    with torch.no_grad():
        y_full = full(x)
        y_split = server(client(x))

    assert y_full.shape == y_split.shape, (y_full.shape, y_split.shape)
    print(f"[OK] split forward works. Output shape = {y_split.shape}")

def freeze_bn(m):
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()

class BoundaryAdapter(nn.Module):
    def __init__(self, C, use_conv=False):
        super().__init__()
        layers = [nn.GroupNorm(1, C, affine=True)]
        if use_conv:
            layers += [nn.Conv2d(C, C, kernel_size=1, bias=False),
        nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class MainServer:
    def __init__(self, base_server_model: nn.Module, num_clients: int, args, device, label_smoothing=0.0, multiple_servers=False):
        self.args = args
        self.device = device
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
        self.memory_dict = {}
        self.wv_history = []
        self.server_lr = args.server_lr

        # "v1" (default, unchanged behaviour): Thapa et al.'s SplitFedV1 --
        # either a persistent per-client server branch (multiple_servers=True)
        # or a single reused branch that every client trains-then-restores
        # against the same snapshot (multiple_servers=False, used purely to
        # save memory); both are FedAvg'd/defended together at round end via
        # aggregate_server_updates_round.
        # "v2": Thapa et al.'s SplitFedV2 -- exactly one server-side model
        # that every client updates in place, sequentially, with no restore
        # and no server-side aggregation ever (only the client side is
        # FedAvg'd). A branch-per-client makes no sense here, so force
        # multiple_servers off regardless of what was passed in.
        self.sfl_variant = getattr(args, "sfl_variant", "v1")
        if self.sfl_variant == "v2":
            multiple_servers = False
        self.multiple_servers = multiple_servers
        self.server_models = []
        self.server_opts = []

        # Create one independent server-side model (branch) per client
        for _ in range(num_clients):
            m = copy.deepcopy(base_server_model).to(device)
            self.server_models.append(m)

            # Prefer SGD to keep memory low with per-client branches
            if args.model == "googlenet":
                opt = torch.optim.Adam(m.parameters(),
                                       lr=args.server_lr,
                                       weight_decay=getattr(args, "wd", 1e-4))
            else:
                opt = torch.optim.SGD(
                    m.parameters(),
                    lr=args.server_lr,
                    momentum=getattr(args, "momentum", 0.0),
                    weight_decay=getattr(args, "weight_decay", 0.0),
                )
            self.server_opts.append(opt)
            if not multiple_servers:
                break  # only one server branch for all clients

        # Optional SafeSplit-style sequential defense on the server model.
        # Only meaningful for sfl_variant="v2" (a single, continuously
        # updated server model) -- see sfl.py's startup validation, which
        # refuses aggr_server="safesplit" under "v1". None (the default)
        # keeps every existing run byte-for-byte unaffected.
        self.safesplit_defense = None
        self._safesplit_step = 0
        if getattr(args, "aggr_server", "avg") == "safesplit":
            self.safesplit_defense = SafeSplitDefense(
                fifo_window=num_clients,
                device=device,
                dct_low_freq_ratio=getattr(args, "safesplit_dct_low_freq_ratio", 0.1),
                reputation_decay=getattr(args, "safesplit_reputation_decay", 0.85),
                reputation_floor=getattr(args, "safesplit_reputation_floor", 0.5),
                gap_min_k=getattr(args, "safesplit_gap_min_k", 2),
            )

    def run_sequential_defense(self, client_id: int):
        """
        SFLv2 only. Call once per client, right after that client's
        local_train_splitfedv1 call has already mutated server_models[0] in
        place (no restore, matching SFLv2) -- analyzes the resulting state
        and rolls the server model back in place if SafeSplit flags this
        step as malicious. No-op whenever aggr_server != "safesplit".
        """
        if self.safesplit_defense is None:
            return True

        model = self.server_models[0]
        self._safesplit_step += 1
        is_benign, rollback_sd = self.safesplit_defense.analyze(
            model.state_dict(), client_id, self._safesplit_step
        )
        if not is_benign:
            model.load_state_dict(rollback_sd)
            # Optimizer momentum/state no longer matches the rolled-back
            # weights -- reinit, mirroring ss_claude.py's Server.run_defense.
            if self.args.model == "googlenet":
                self.server_opts[0] = torch.optim.Adam(
                    model.parameters(),
                    lr=self.args.server_lr,
                    weight_decay=getattr(self.args, "wd", 1e-4),
                )
            else:
                self.server_opts[0] = torch.optim.SGD(
                    model.parameters(),
                    lr=self.args.server_lr,
                    momentum=getattr(self.args, "momentum", 0.0),
                    weight_decay=getattr(self.args, "weight_decay", 0.0),
                )
        return is_benign

    def smooth_cross_entropy(self, logits, targets, num_classes):
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.label_smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        return torch.mean(torch.sum(-true_dist * F.log_softmax(logits, dim=1), dim=1))

    def train_on_smashed(self, client_id: int, smashed: torch.Tensor, y: torch.Tensor):
        """
        smashed must be a LEAF tensor with requires_grad=True on the server side.
        Returns: dL/dsmashed
        """
        if not self.multiple_servers:
            model = self.server_models[0]
            opt = self.server_opts[0]
        else:
            model = self.server_models[client_id]
            opt = self.server_opts[client_id]
        model.train()
        model.apply(freeze_bn)
        smashed = smashed.to(self.device)
        y = y.to(self.device)

        # Ensure grad on smashed will be retained
        if not smashed.requires_grad:
            smashed.requires_grad_(True)
        smashed.retain_grad()

        opt.zero_grad(set_to_none=True)

        logits = model(smashed)
        if self.label_smoothing > 0.0:
            num_classes = next(
                m.out_features for m in self.server_models[0].modules()
                if isinstance(m, torch.nn.Linear)
            )
            loss = self.smooth_cross_entropy(logits, y, num_classes=num_classes)
        else:
            loss = self.criterion(logits, y)
        loss.backward()

        dfx = smashed.grad
        # print("server dfx norm:", dfx.norm().item())
        if dfx is None:
            raise RuntimeError("smashed.grad is None — ensure smashed.requires_grad=True and retain_grad() is called.")

        # dfx = dfx.detach()  # feedback tensor to send to client
        # max_feedback_norm = 1.0 # try 0.5–2.0
        # n = dfx.norm()
        # if torch.isfinite(n) and n > max_feedback_norm:
        #     dfx = dfx * (max_feedback_norm / (n + 1e-12))
        # dfx = dfx.detach()

        # gradient w.r.t smashed activations
        dfx = smashed.grad.detach().clone()
        # print("mean|dfx|", smashed.grad.abs().mean().item())
        # with torch.no_grad():
        #     p0 = next(model.parameters())
        #     gnorm = p0.grad.abs().mean().item() if p0.grad is not None else -1
        # print("mean|server_grad|", gnorm)

        opt.step()

        # optional accuracy
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == y).float().mean() * 100.0

        return dfx, loss.detach(), acc.detach()

    @torch.no_grad()
    def eval_on_smashed(self, client_id: int, smashed: torch.Tensor, y: torch.Tensor):
        if not self.multiple_servers:
            model = self.server_models[0]
        else:
            model = self.server_models[client_id]
        model.eval()

        smashed = smashed.to(self.device)
        y = y.to(self.device)

        logits = model(smashed)
        if self.label_smoothing > 0.0:
            num_classes = next(
                m.out_features for m in self.server_models[0].modules()
                if isinstance(m, torch.nn.Linear)
            )
            loss = self.smooth_cross_entropy(logits, y, num_classes=num_classes)
        else:
            loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean() * 100.0
        return loss.detach(), acc.detach()



    @staticmethod
    def _state_dict_to_vec(state_dict, keys, device):
        flat = []
        for k in keys:
            flat.append(state_dict[k].detach().to(device).reshape(-1))
        return torch.cat(flat)

    @staticmethod
    def _update_dict_to_vec(update_dict, keys, device):
        flat = []
        for k in keys:
            t = update_dict[k].detach().to(device)
            flat.append(t.reshape(-1))
        return torch.cat(flat)

    @staticmethod
    def _vec_to_update_dict(vec, ref_state_dict, keys):
        out = {}
        offset = 0
        for k in keys:
            numel = ref_state_dict[k].numel()
            out[k] = vec[offset:offset+numel].view_as(ref_state_dict[k])
            offset += numel
        return out

    @staticmethod
    def _cos_sim(a, b, eps=1e-12):
        return torch.dot(a, b) / (a.norm() * b.norm() + eps)

    @staticmethod
    def _sgn(v):
        # returns {-1,0,+1} float tensor
        return torch.sign(v)

    @staticmethod
    def _topk_mask(v, k):
        # mask of shape [d] with 1 at top-k |v| entries
        d = v.numel()
        k = int(max(1, min(k, d)))
        _, idx = torch.topk(v.abs(), k, largest=True, sorted=False)
        mask = torch.zeros(d, device=v.device, dtype=v.dtype)
        mask[idx] = 1.0
        return mask

    @staticmethod
    def _mz_scores(x, eps=1e-12):
        # Median-based Z-score variant used in your text: (xi - median)/std
        med = x.median()
        std = x.std(unbiased=False) + eps
        return (x - med) / std

    @staticmethod
    def _clip_by_l2(v, c, eps=1e-12):
        # per-vector clipping to L2 norm c
        n = v.norm() + eps
        if n > c:
            return v * (c / n)
        return v

    def aggregate_server_updates_round(
        self,
        client_ids,
        agent_server_updates,   # dict: agent_id -> update_dict (same keys/shapes as server state_dict)
        agent_data_sizes,       # dict: agent_id -> n_data (for FedAvg weighting)
        rnd=None
    ):
        """
        Aggregate (and optionally defend) server-side updates BEFORE applying them.
        - Uses self.args.aggr_server in {"avg","alignins"} (others fall back to avg).
        - Applies the aggregated delta to a global server model, then syncs all branches.

        Expected: agent_server_updates contains DELTAS for server params (e.g., -lr * grad).
        """

        if self.sfl_variant == "v2":
            # SplitFedV2: the single server-side model was already updated in
            # place, client after client, inside each Agent's training call
            # (no restore happened, so there's nothing to aggregate here --
            # matching Thapa et al., who never run FedAvg/a defense on the
            # server side in V2, only on the client side).
            return

        if len(agent_server_updates) == 0:
            return

        device = self.device
        # Choose a "global" server model reference (branch 0) and sync all branches from it after update
        global_model = self.server_models[0]
        global_sd = global_model.state_dict()
        keys = list(global_sd.keys())

        # Build weighted update vectors
        participating = [i for i in client_ids if i in agent_server_updates]
        if len(participating) == 0:
            return

        round_total = sum(agent_data_sizes[i] for i in participating)
        upd_vecs = {}
        weights = {}

        for i in participating:
            w = agent_data_sizes[i] / max(1, round_total)
            v = self._update_dict_to_vec(agent_server_updates[i], keys, device=device)
            upd_vecs[i] = v
            weights[i] = float(w)
        V = torch.stack([upd_vecs[i] for i in participating], dim=0) 
        n_clients = V.shape[0]
        agent_updates_dict = {}
        for i in client_ids:
            if i in agent_server_updates:
                agent_updates_dict[i] = self._update_dict_to_vec(
                    agent_server_updates[i], keys, device=device
                )
        cur_global_params = self._state_dict_to_vec(global_sd, keys, device=device)
        # --- Aggregation rules ---
        defense_start_round = getattr(self.args, 'defense_start_round', 1)
        if defense_start_round > 1 and rnd < defense_start_round:
            print(f'MS: not yet defended round:{rnd}, starts at {defense_start_round}')
            # FedAvg-style weighted mean on deltas (no filtering)
            num = None
            den = 0.0
            for i in participating:
                v = upd_vecs[i]
                w = weights[i]
                if num is None:
                    num = w * v
                else:
                    num += w * v
                den += w
            aggregated = num / max(den, 1e-12)
        elif getattr(self.args, "aggr_server", "avg") == "alignins":
            print(f'MS: alignins defended round:{rnd}, started at {defense_start_round}')
            # AlignIns-style filtering on server deltas (TDA + MPSA + MZ + clipping)

            # Reference vector for TDA: current global server parameters
            theta = self._state_dict_to_vec(global_sd, keys, device=device)

            ids = participating
            V = torch.stack([upd_vecs[i] for i in ids], dim=0)  # [n, d]
            n, d = V.shape

            # 1) TDA scores (cosine similarity update vs global params)
            tda = torch.tensor([self._cos_sim(upd_vecs[i], theta).item() for i in ids],
                               device=device, dtype=torch.float32)

            # 2) Principal sign p (majority vote per coordinate on sign)
            S = self._sgn(V)  # [n, d]
            p = self._sgn(S.sum(dim=0))  # [d]

            # 3) MPSA scores (Top-k sign alignment to principal sign)
            # choose k: use args.dense_ratio if you have it; else 1% of dims
            dense_ratio = float(getattr(self.args, "dense_ratio", 0.01))
            k = int(max(1, min(d, round(d * dense_ratio))))

            mpsa_list = []
            for i in range(n):
                vi = V[i]
                mask = self._topk_mask(vi, k)  # [d]
                diff = (self._sgn(vi) - p) * mask  # keep only top-k coords
                # count mismatched coords among top-k:
                mism = (diff != 0).sum().float()
                rho = 1.0 - (mism / float(k))
                mpsa_list.append(rho)
            mpsa = torch.stack(mpsa_list).to(device).float()  # [n]

            # 4) MZ outlier filtering
            lam_c = float(getattr(self.args, "lambda_c", 2.5))
            lam_s = float(getattr(self.args, "lambda_s", 2.5))

            mz_tda = self._mz_scores(tda)
            mz_mpsa = self._mz_scores(mpsa)

            keep = (mz_tda.abs() <= lam_c) & (mz_mpsa.abs() <= lam_s)

            kept_ids = [ids[j] for j in range(n) if bool(keep[j].item())]

            # Safety: if filtering is too aggressive, keep at least a majority of participants
            # (AlignIns paper uses a condition like |S| >= n - 2m; here we enforce >= ceil(n/2))
            min_keep = max(1, (n + 1) // 2)
            if len(kept_ids) < min_keep:
                # fallback: keep those closest to median in (tda,mpsa) combined score
                med_tda = tda.median()
                med_mpsa = mpsa.median()
                dist = (tda - med_tda).abs() + (mpsa - med_mpsa).abs()
                order = torch.argsort(dist)  # smallest first
                kept_ids = [ids[int(order[j])] for j in range(min_keep)]

            # 5) Clip kept updates and do weighted average
            c = float(getattr(self.args, "clip_c", getattr(self.args, "c", 5.0)))
            num = torch.zeros(d, device=device)
            den = 0.0
            for i in kept_ids:
                v = self._clip_by_l2(upd_vecs[i], c)
                w = weights[i]
                num += w * v
                den += w
            aggregated = num / max(den, 1e-12)
        elif getattr(self.args, "aggr_server", "avg") == "mmetric":
            print(f'MS: mmetric defended round:{rnd}, started at {defense_start_round}')
            aggregated = self.agg_mul_metric(agent_updates_dict, global_model, cur_global_params)
        elif getattr(self.args, "aggr_server", "avg") == "foolsgold":
            print(f'MS: foolsgold defended round:{rnd}, started at {defense_start_round}')
            aggregated = self.agg_foolsgold(agent_updates_dict)
        elif getattr(self.args, "aggr_server", "avg") == "signguard":
            print(f'MS: signguard defended round:{rnd}, started at {defense_start_round}')
            aggregated = self.agg_signguard(agent_updates_dict)
        elif getattr(self.args, "aggr_server", "avg") == "mkrum":
            print(f'MS: mkrum defended round:{rnd}, started at {defense_start_round}')
            aggregated = self.agg_mkrum(agent_updates_dict)
        elif getattr(self.args, "aggr_server", "avg") == "rfa":
            print(f'MS: rfa defended round:{rnd}, started at {defense_start_round}')
            aggregated = self.agg_rfa(agent_updates_dict)
        elif getattr(self.args, "aggr_server", "avg") == "feddlad":
            print(f'MS: feddlad defended round:{rnd}, started at {defense_start_round}')
            aggregated = self.agg_feddlad(agent_updates_dict)
        elif getattr(self.args, "aggr_server", "avg") == "rlr":
            # 1. Get the robustness mask from your function
            # Note: we ignore the first return if it contains the raw sum_of_signs
            _, rlr_mask = self.compute_robustLR(agent_updates_dict)
            
            # 2. Compute the standard average of updates
            base_avg = torch.stack(list(agent_updates_dict.values())).mean(dim=0)
            
            # 3. Apply the mask and the server learning rate
            # This ensures that only parameters with high agreement get updated
            aggregated = base_avg * rlr_mask 
        
        else:
            print(f'MS: not defended round:{rnd}, should have started at {defense_start_round}')
            # FedAvg-style weighted mean on deltas (no filtering)
            num = None
            den = 0.0
            for i in participating:
                v = upd_vecs[i]
                w = weights[i]
                if num is None:
                    num = w * v
                else:
                    num += w * v
                den += w
            aggregated = num / max(den, 1e-12)

        # --- Apply aggregated delta to global server model ---
        new_sd = copy.deepcopy(global_sd)
        delta_dict = self._vec_to_update_dict(aggregated, new_sd, keys)
        for k in keys:
            new_sd[k] = new_sd[k].to(device) + delta_dict[k].to(new_sd[k].device)

        global_model.load_state_dict(new_sd)

        # --- Sync all branches to the updated global ---
        for m in self.server_models:
            m.load_state_dict(global_model.state_dict())
    def agg_avg(self, agent_updates_dict):
        """
        Revised FedAvg for flat tensors (Server-side updates).
        """
        if not agent_updates_dict:
            return None

        # Get the list of tensors
        updates = list(agent_updates_dict.values())
        
        # Simple tensor summation and division
        # torch.stack creates a new dimension, then we mean across it
        aggregated_tensor = torch.stack(updates).mean(dim=0)
        
        return aggregated_tensor

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
        # Calculate the sum of signs across all clients
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        
        # Create a mask: 1 where clients agree, 0 where they diverge
        # self.args.theta is typically set to a value like 3 or 5
        mask = torch.zeros_like(sm_of_signs)
        mask[sm_of_signs >= self.args.theta] = 1.0
        
        # If you want to use the server_lr here:
        # We return the mask. You apply the LR in the aggregation step.
        return mask.to(self.args.device), mask

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
    def agg_feddlad(self, agent_updates_dict):
        """
        FedDLAD Implementation
        Interface: agent_updates_dict {id: tensor} -> returns aggregated_tensor
        """
        agent_ids = list(agent_updates_dict.keys())
        # Convert tensors to numpy for structural analysis (SKLearn)
        client_updates_np = [update.detach().cpu().numpy().flatten() for update in agent_updates_dict.values()]
        updates_stack_np = np.array(client_updates_np)
        
        # --- Layer 1: Structural Connectivity Analysis (COF/LOF) ---
        # COF identifies outliers based on local density and connectivity
        # We use LOF with a high neighbor count to approximate structural connectivity
        num_clients = len(agent_ids)
        n_neighbors = max(2, num_clients // 4)
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False, metric='cosine')
        
        # predict returns 1 for inliers, -1 for outliers
        outlier_labels = clf.fit_predict(updates_stack_np)
        
        trusted_ids = [agent_ids[i] for i, label in enumerate(outlier_labels) if label == 1]
        untrusted_ids = [agent_ids[i] for i, label in enumerate(outlier_labels) if label == -1]

        # If Layer 1 fails to find anyone, fallback to simple average
        if not trusted_ids:
            return torch.mean(torch.stack(list(agent_updates_dict.values())), dim=0)

        # --- Layer 2: IQR Refinement (Inside the Trusted Group) ---
        trusted_updates = torch.stack([agent_updates_dict[idx] for idx in trusted_ids])
        norms = torch.norm(trusted_updates, dim=1)
        
        q1 = torch.quantile(norms, 0.25)
        q3 = torch.quantile(norms, 0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter trusted clients who are "insider threats" (norms too high/low)
        refined_trusted_ids = []
        for idx in trusted_ids:
            n = torch.norm(agent_updates_dict[idx])
            if lower_bound <= n <= upper_bound:
                refined_trusted_ids.append(idx)
                
        if not refined_trusted_ids:
            refined_trusted_ids = trusted_ids # Fallback to layer 1 results

        # Generate reference direction from refined trusted group
        reference_update = torch.mean(torch.stack([agent_updates_dict[idx] for idx in refined_trusted_ids]), dim=0)

        # --- Pardon Module: Re-include misclassified benign clients ---
        final_indices = copy.deepcopy(refined_trusted_ids)
        for idx in untrusted_ids:
            # Check cosine similarity between "untrusted" and the "refined trusted" reference
            sim = torch.nn.functional.cosine_similarity(
                agent_updates_dict[idx].unsqueeze(0), 
                reference_update.unsqueeze(0)
            )
            # If the untrusted client is very similar to the trusted direction, pardon it
            if sim.item() > 0.8: 
                final_indices.append(idx)

        # --- Final Weighted Aggregation ---
        # FedDLAD typically uses simple averaging of the "cleared" group
        final_updates = [agent_updates_dict[idx] for idx in final_indices]
        aggregated_model = torch.mean(torch.stack(final_updates), dim=0)

        print(f"[FedDLAD] Trusted: {len(refined_trusted_ids)} | Pardoned: {len(final_indices) - len(refined_trusted_ids)} | Total: {len(final_indices)}/{num_clients}")

        return aggregated_model
   
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

    def fedavg_server_branches(self, client_ids):
        """Optional SplitFedV1 step: average server branches then sync all branches."""
        if not self.multiple_servers:
            return  # only one branch, nothing to do
        state_dicts = [self.server_models[i].state_dict() for i in client_ids]
        avg = copy.deepcopy(state_dicts[0])
        for k in avg:
            for sd in state_dicts[1:]:
                avg[k] += sd[k]
            avg[k] = avg[k] / len(state_dicts)

        for m in self.server_models:
            m.load_state_dict(avg)
from resnet import ResNet9, ResNet18
import torchvision.models as models
import torch.nn as nn
def get_model(data, args):
    if data == 'cifar10':
        model = ResNet9(3,num_classes=10, args=args)
    elif data == "mnist":
        model = ResNet18(1,num_classes=10, args=args)
    elif data == 'tinyimagenet':
        model = get_resnet18_64x64()

    return model
         

def get_resnet18_64x64(num_classes=200):
    """
    Returns a ResNet-18 model modified for 64x64 input images and a specific number of output classes.

    Args:
        num_classes (int): The number of output classes. Default is 200.

    Returns:
        model (torch.nn.Module): Modified ResNet-18 model.
    """
    # Load the ResNet-18 model pretrained on ImageNet
    model = models.resnet18(pretrained=False)

    # Modify the first convolutional layer to handle 64x64 images
    model.conv1 = nn.Conv2d(
        in_channels=3, 
        out_channels=64, 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        bias=False
    )
    
    # Adjust the max pooling layer to fit smaller image sizes
    model.maxpool = nn.Identity()

    # Modify the fully connected layer to output num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
