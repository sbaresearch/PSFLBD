import torch
import torch.nn as nn
from torchvision import models
import copy
import logging
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


def get_resnet18_custom(data: str = "tinyimagenet", num_classes: int = 200) -> nn.Module:
    """
    ResNet-18 modified for CIFAR-like inputs (3x32x32) and MNIST (1x28x28):
      - conv1: 3x3 stride 1
      - maxpool: Identity (common for small images)
      - fc: adjusted to num_classes
    """
    model = models.resnet18(weights=None)

    in_channels = 1 if data == "mnist" or data == "fmnist" else 3

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
    full_model = get_resnet18_custom(data=data, num_classes=num_classes)

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


def get_resnet9_split(data: str, args) -> tuple[nn.Module, nn.Module, nn.Module]:
    """
    Same pattern: build ONE full model, then split.
    Adjust this split based on your ResNet9 definition.
    """
    from resnet import ResNet9

    in_channels = 1 if data == "mnist" or data == "fmnist" else 3
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
        return get_resnet18_split(data=data, num_classes=10,split_point = args.split_point)

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
    elif data in ("cifar10", "cifar100"):
        x = torch.randn(4, 3, 32, 32, device=device)
    elif data == "tinyimagenet":
        x = torch.randn(4, 3, 64, 64, device=device)
    else:
        x = torch.randn(4, 3, 32, 32, device=device)

    with torch.no_grad():
        y_full = full(x)
        y_split = server(client(x))

    assert y_full.shape == y_split.shape, (y_full.shape, y_split.shape)
    logging.info(f"[OK] split forward works. Output shape = {y_split.shape}")

def freeze_bn(m):
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()

class MainServer:
    def __init__(self, base_server_model: nn.Module, num_clients: int, args, device):
        self.args = args
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(device)

        self.server_models = []
        self.server_opts = []

        # Create one independent server-side model (branch) per client
        for _ in range(num_clients):
            m = copy.deepcopy(base_server_model).to(device)
            self.server_models.append(m)

            # Prefer SGD to keep memory low with per-client branches
            opt = torch.optim.SGD(
                m.parameters(),
                lr=args.server_lr,
                momentum=getattr(args, "momentum", 0.0),
                weight_decay=getattr(args, "weight_decay", 0.0),
            )
            self.server_opts.append(opt)

    def train_on_smashed(self, client_id: int, smashed: torch.Tensor, y: torch.Tensor):
        """
        smashed must be a LEAF tensor with requires_grad=True on the server side.
        Returns: dL/dsmashed
        """
        model = self.server_models[client_id]
        opt = self.server_opts[client_id]
        model.train()
        model.apply(freeze_bn)
        smashed = smashed.detach().to(self.device).requires_grad_(True)
        y = y.to(self.device)

        opt.zero_grad(set_to_none=True)
        # if self.args.debug:
        #     logging.info(f"\n[Server] Client {client_id} sent activations: shape={list(smashed.shape)}, "
        #         f"mean={smashed.mean().item():.4f}, std={smashed.std().item():.4f}")
        logits = model(smashed)
        loss = self.criterion(logits, y)
        loss.backward()

        # gradient w.r.t smashed activations
        dfx = smashed.grad.detach().clone()
        # print("mean|dfx|", smashed.grad.abs().mean().item())
        # with torch.no_grad():
        #     p0 = next(model.parameters())
        #     gnorm = p0.grad.abs().mean().item() if p0.grad is not None else -1
        # print("mean|server_grad|", gnorm)
        # if self.args.debug:
        #     logging.info(f"[Server] Generated dL/dfx: mean={dfx.abs().mean().item():.6f}, "
        #         f"max={dfx.abs().max().item():.6f}")
        opt.step()

        # optional accuracy
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == y).float().mean() * 100.0

        return dfx, loss.detach(), acc.detach()

    @torch.no_grad()
    def eval_on_smashed(self, client_id: int, smashed: torch.Tensor, y: torch.Tensor):
        model = self.server_models[client_id]
        model.eval()

        smashed = smashed.to(self.device)
        y = y.to(self.device)

        logits = model(smashed)
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
    ):
        """
        Aggregate (and optionally defend) server-side updates BEFORE applying them.
        - Uses self.args.aggr_server in {"avg","alignins"} (others fall back to avg).
        - Applies the aggregated delta to a global server model, then syncs all branches.

        Expected: agent_server_updates contains DELTAS for server params (e.g., -lr * grad).
        """

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

        # --- Aggregation rules ---
        if getattr(self.args, "aggr_server", "avg") == "alignins":
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

        else:
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



    def fedavg_server_branches(self, client_ids):
        """Optional SplitFedV1 step: average server branches then sync all branches."""
        state_dicts = [self.server_models[i].state_dict() for i in client_ids]
        avg = copy.deepcopy(state_dicts[0])
        for k in avg:
            for sd in state_dicts[1:]:
                avg[k] += sd[k]
            avg[k] = avg[k] / len(state_dicts)

        for m in self.server_models:
            m.load_state_dict(avg)
from resnet import ResNet9, ResNet18
from vgg import VGG
import torchvision.models as models
import torch.nn as nn
def get_model(data, args):
    if data == 'cifar10':
        model = ResNet9(3,num_classes=10, args=args)
    elif data == "mnist":
        model = ResNet18(1,num_classes=10, args=args)
    elif data == 'cifar100':
        model = VGG('VGG9',num_classes=100)
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

    return model