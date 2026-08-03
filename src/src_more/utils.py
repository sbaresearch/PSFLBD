import copy
import logging
import os
import glob, csv
from PIL import Image

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from math import floor
from collections import defaultdict
import random
import math
import time
from shutil import copyfile
from torchvision.datasets.utils import download_and_extract_archive

def setup_logging(args):
    # Create the directory for logs
    log_dir = f"outputs/{args.data}_{args.attack}_{args.aggr_server}_{args.aggr}"
    # Keep v1's path exactly as before (default); only tag the dir for v2 so
    # the two variants never collide/overwrite each other's results.csv when
    # run with the same data/attack/aggr combo.
    if getattr(args, "sfl_variant", "v1") != "v1":
        log_dir = f"{log_dir}_{args.sfl_variant}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "main.log")

    # Define the format
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    # 1. THE FILE HANDLER (Keep this)
    file_handler = logging.FileHandler(log_path, mode = 'w')
    file_handler.setFormatter(log_formatter)

    # 2. GET THE ROOT LOGGER
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid double-logging
    root_logger.handlers = []

    # 3. ADD ONLY THE FILE HANDLER
    root_logger.addHandler(file_handler)

    # Note: Do NOT add a logging.StreamHandler() here
    
    return log_dir

class DatasetSplit(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """

    def __init__(self, dataset, idxs, runtime_poison=False, args=None, client_id=-1, modify_label=True):
        self.dataset = dataset
        self.idxs = idxs
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])
        self.runtime_poison = runtime_poison
        self.args = args
        self.client_id = client_id
        self.modify_label = modify_label
        if client_id == -1:
            poison_frac = 1
        elif client_id < self.args.num_corrupt:
            poison_frac = self.args.poison_frac
        else:
            poison_frac = 0
        self.poison_sample = {}
        self.poison_idxs = []
        if runtime_poison and poison_frac > 0:
            self.poison_idxs = random.sample(self.idxs, floor(poison_frac * len(self.idxs)))
            for idx in self.poison_idxs:
                self.poison_sample[idx] = add_pattern_bd(copy.deepcopy(self.dataset[idx][0]), None, args.data,
                                                         pattern_type=args.pattern_type, agent_idx=client_id,
                                                         attack=args.attack)
                # plt.imshow(self.poison_sample[idx].permute(1, 2, 0))
                # plt.show()

    def classes(self):
        return torch.unique(self.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # print(target.type())
        if self.idxs[item] in self.poison_idxs:
            inp = self.poison_sample[self.idxs[item]]
            if self.modify_label:
                target = self.args.target_class
            else:
                target = self.dataset[self.idxs[item]][1]
        else:
            inp, target = self.dataset[self.idxs[item]]

        return inp, target


class DatasetSplit_new(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """

    def __init__(self, dataset, backdoor_dataset, benign_idx, backdoor_idx, idxs, runtime_poison=False, args=None, client_id=-1, modify_label=True):
        self.dataset = dataset
        self.backdoor_dataset = backdoor_dataset
        self.backdoor_dataset.targets = torch.ones_like(self.backdoor_dataset.targets)
        self.benign_idx = benign_idx
        self.backdoor_idx = backdoor_idx
        self.idxs = idxs
        self.targets = torch.Tensor(self.dataset.targets.float())
        self.runtime_poison = runtime_poison
        self.args = args
        self.client_id = client_id
        self.modify_label = modify_label
        if client_id == -1:
            poison_frac = 1
        elif client_id < self.args.num_corrupt:
            poison_frac = self.args.poison_frac
        else:
            poison_frac = 0
        self.poison_sample = {}
        self.poison_idxs = []
                # plt.imshow(self.poison_sample[idx].permute(1, 2, 0))
                # plt.show()

    def classes(self):
        return torch.unique(self.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # print(target.type())
        # print(item)
        # print(self.benign_idx)
        # print(self.backdoor_idx)
        if item not in self.benign_idx:
            inp, target = self.backdoor_dataset[self.idxs[item]]
            target = torch.ones_like(target)
            # print('yes')
        else:
            inp, target = self.dataset[self.idxs[item]]
        # inp, target = self.dataset[self.idxs[item]]

        return inp, target


def distribute_data_dirichlet(dataset, args):
    # sort labels
    labels_sorted = dataset.targets.sort()
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    labels_dict = defaultdict(list)

    for k, v in class_by_labels:
        labels_dict[k].append(v)
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    N = len(labels_sorted[1])
    K = len(labels_dict)
    logging.info((N, K))
    client_num = args.num_agents

    min_size = 0
    while min_size < 10:
        idx_batch = [[] for _ in range(client_num)]
        for k in labels_dict:
            idx_k = labels_dict[k]

            # get a list of batch indexes which are belong to label k
            np.random.shuffle(idx_k)
            # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
            # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
            proportions = np.random.dirichlet(np.repeat(args.beta, client_num))

            # get the index in idx_k according to the dirichlet distribution
            proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            # generate the batch list for each client
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    # distribute data to users
    dict_users = defaultdict(list)
    for user_idx in range(args.num_agents):
        dict_users[user_idx] = idx_batch[user_idx]
        np.random.shuffle(dict_users[user_idx])

    num = [ [ 0 for k in range(K) ] for i in range(client_num)]
    for k in range(K):
        for i in dict_users:
            num[i][k] = len(np.intersect1d(dict_users[i], labels_dict[k]))
    # logging.info(num)
    # print(dict_users)
    # def intersection(lst1, lst2):
    #     lst3 = [value for value in lst1 if value in lst2]
    #     return lst3
    # client_label_num = [len(intersection (dict_users[i], dict_users[i+1] )) for i in range(args.num_agents)]

    for each_client, id_ in zip(num, range(len(num))):
        logging.info('client:%d, distribution: %s' % (id_, each_client))
    return dict_users


def distribute_data(dataset, args, n_classes=10):
    # logging.info(dataset.targets)
    # logging.info(dataset.classes)
    class_per_agent = n_classes

    if args.num_agents == 1:
        return {0: range(len(dataset))}

    def chunker_list(seq, size):
        return [seq[i::size] for i in range(size)]

    # sort labels
    labels_sorted = torch.tensor(dataset.targets).sort()
    # print(labels_sorted)
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)

    # split indexes to shards
    shard_size = len(dataset) // (args.num_agents * class_per_agent)
    slice_size = (len(dataset) // n_classes) // shard_size
    for k, v in labels_dict.items():
        labels_dict[k] = chunker_list(v, slice_size)
    hey = copy.deepcopy(labels_dict)
    # distribute shards to users
    dict_users = defaultdict(list)
    for user_idx in range(args.num_agents):
        class_ctr = 0
        for j in range(0, n_classes):
            if class_ctr == class_per_agent:
                break
            elif len(labels_dict[j]) > 0:
                dict_users[user_idx] += labels_dict[j][0]
                del labels_dict[j % n_classes][0]
                class_ctr += 1
        np.random.shuffle(dict_users[user_idx])
    # num = [ [ 0 for k in range(n_classes) ] for i in range(args.num_agents)]
    # for k in range(n_classes):
    #     for i in dict_users:
    #         num[i][k] = len(np.intersect1d(dict_users[i], hey[k]))
    # logging.info(num)
    # logging.info(args.num_agents)
    # def intersection(lst1, lst2):
    #     lst3 = [value for value in lst1 if value in lst2]
    #     return lst3
    # logging.info( len(intersection (dict_users[0], dict_users[1] )))

    return dict_users
class FEMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, transform=None):
        self.data = X
        self.targets = Y
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx], self.targets[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class GTSRBFallback(Dataset):
    TRAIN_URL = "https://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip"
    TEST_URL = "https://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip"

    def __init__(self, root, split='train', download=True, transform=None, target_transform=None):
        super().__init__()
        assert split in ('train', 'test')
        self.root = os.path.join(root, 'GTSRB')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if download:
            os.makedirs(self.root, exist_ok=True)
            # Download/extract if needed
            train_flag = os.path.join(self.root, 'Final_Training', 'Images')
            test_flag = os.path.join(self.root, 'Final_Test', 'Images')
            if not os.path.isdir(train_flag):
                download_and_extract_archive(self.TRAIN_URL, download_root=self.root)
            if not os.path.isdir(test_flag):
                download_and_extract_archive(self.TEST_URL, download_root=self.root)

        self.samples = []
        if split == 'train':
            train_dir = os.path.join(self.root, 'Final_Training', 'Images')
            # Class folders are 00000, 00001, ..., 00042
            class_dirs = sorted(d for d in glob.glob(os.path.join(train_dir, '*')) if os.path.isdir(d))
            for cd in class_dirs:
                label = int(os.path.basename(cd))
                for img_path in glob.glob(os.path.join(cd, '*.*')):
                    self.samples.append((img_path, label))
        else:
            test_dir = os.path.join(self.root, 'Final_Test', 'Images')
            # Find the test ground-truth CSV (delimiter=';')
            csv_candidates = glob.glob(os.path.join(self.root, '**', 'GT-final_test.csv'), recursive=True)
            if not csv_candidates:
                raise FileNotFoundError("GT-final_test.csv not found after extraction.")
            gt_csv = csv_candidates[0]
            fname_to_label = {}
            with open(gt_csv, newline='') as f:
                reader = csv.reader(f, delimiter=';')
                header = next(reader, None)
                # Expect header with 'ClassId' and 'Filename'
                for row in reader:
                    # Columns: Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
                    fname = row[0]
                    class_id = int(row[-1])
                    fname_to_label[fname] = class_id
            for img_path in glob.glob(os.path.join(test_dir, '*.*')):
                fname = os.path.basename(img_path)
                if fname in fname_to_label:
                    self.samples.append((img_path, fname_to_label[fname]))
            # Keep deterministic order
            self.samples.sort(key=lambda x: x[0])

        self.targets = torch.tensor([t for _, t in self.samples], dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

def _extract_targets(ds):
    if hasattr(ds, "targets"):
        t = ds.targets
    elif hasattr(ds, "labels"):
        t = ds.labels
    elif hasattr(ds, "samples"):
        t = [s[1] for s in ds.samples]
    else:
        # Fallback (slow): iterate once
        t = [ds[i][1] for i in range(len(ds))]
    return torch.tensor(t, dtype=torch.long)

def get_datasets(data):
    """ returns train and test datasets """
    train_dataset, test_dataset = None, None
    data_dir = './data'

    if data == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    if data == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    elif data == 'fedemnist':
        train_dir = '../data/Fed_EMNIST/fed_emnist_all_trainset.pt'
        test_dir = '../data/Fed_EMNIST/fed_emnist_all_valset.pt'
        train_dataset = torch.load(train_dir)
        test_dataset = torch.load(test_dir)
    elif data == 'femnist':
        transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_path = './data/femnist/train.pt'
        test_path = './data/femnist/test.pt'

        X_train, Y_train = torch.load(train_path)
        X_test, Y_test = torch.load(test_path)

        train_dataset = FEMNISTDataset(X_train, Y_train, transform)
        test_dataset = FEMNISTDataset(X_test, Y_test, transform)

    elif data == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(
            test_dataset.targets)
    elif data == 'cifar100':
        transform = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                             std=[0.2675, 0.2565, 0.2761])])
        valid_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                                   std=[0.2675, 0.2565, 0.2761])])
        train_dataset = datasets.CIFAR100(data_dir,
                                          train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(data_dir,
                                         train=False, download=True, transform=valid_transform)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(
            test_dataset.targets)
    elif data == "tinyimagenet":
        _data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor()
            ]),
            'test': transforms.Compose([
                transforms.ToTensor()
            ]),
        }
        _data_dir = './data/tiny-imagenet-200/'
        train_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'train'),
                                             _data_transforms['train'])
        # print(train_dataset[0][0].shape)
        test_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'test'),
                                            _data_transforms['test'])
        train_dataset.targets = torch.tensor(train_dataset.targets)
        test_dataset.targets = torch.tensor(test_dataset.targets)
    elif data == 'gtsrb':
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        img_size = 32  # try 64 or 96 for good accuracy/cost tradeoff; 224 for best accuracy
        transform_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ], p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Use torchvision.datasets.GTSRB if available; otherwise fallback
        if hasattr(datasets, 'GTSRB'):
            train_dataset = datasets.GTSRB(root=data_dir, split='train', download=True, transform=transform_train)
            test_dataset = datasets.GTSRB(root=data_dir, split='test', download=True, transform=transform_test)
        else:
            train_dataset = GTSRBFallback(root=data_dir, split='train', download=False, transform=transform_train)
            test_dataset = GTSRBFallback(root=data_dir, split='test', download=False, transform=transform_test)

        
        # Make sure .targets is a torch.long tensor across versions
        train_dataset.targets = _extract_targets(train_dataset)
        test_dataset.targets = _extract_targets(test_dataset)
    elif data == 'ets':
        # European Traffic Signs (ETS)
        ets_root = './data/ETS'
        train_dir = os.path.join(ets_root, 'Training')
        test_dir = os.path.join(ets_root, 'Testing')


        # Many traffic-sign datasets have varied image sizes; resize to a fixed size.
        # Avoid random horizontal flip (flipping can change sign meaning).
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

        # Ensure labels are tensors for consistency with your other loaders
        train_dataset.targets = torch.tensor(train_dataset.targets, dtype=torch.long)
        test_dataset.targets = torch.tensor(test_dataset.targets, dtype=torch.long)
    return train_dataset, test_dataset


def get_loss_n_accuracy(model, criterion, data_loader, args, round, num_classes=10, sent140_flag=False):
    """ Returns the loss and total accuracy, per class accuracy on the supplied data loader """

    # disable BN stats during inference
    model.eval()
    total_loss, correctly_labeled_samples = 0, 0
    # confusion_matrix = torch.zeros(num_classes, num_classes)
    # not_correct_samples = []
    # forward-pass to get loss and predictions of the current batch
    all_labels = []
    # if round % 20 == 0:
    #     def hook(module, fea_n, fea_out):
    #         representation.append(fea_out.detach().cpu())
    #         return None
    #     for (name, module) in model.named_modules():
    #         if name == layer_name:
    #             handle = module.register_forward_hook(hook=hook)

    for _, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device=args.device, non_blocking=True), \
                         labels.to(device=args.device, non_blocking=True)
        # compute the total loss over minibatch
        if sent140_flag:
            labels = torch.ones_like(labels)
        outputs = model(inputs)
        if args.data == 'sen140':
            avg_minibatch_loss = criterion(outputs.float(), labels.float())
        else:
            avg_minibatch_loss = criterion(outputs, labels)

        total_loss += avg_minibatch_loss.item() * outputs.shape[0]

        # get num of correctly predicted inputs in the current batch
        if args.data == 'sen140':
            # print(outputs.shape)
            # print(outputs)
            pred_labels = outputs.squeeze() > 0.5
            # _, pred_labels = torch.max(outputs, 1)
            # print(pred_labels)
        else:
            _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        all_labels.append(labels.cpu().view(-1))
        # correct_inputs = labels[torch.nonzero(torch.eq(pred_labels, labels) == 0).squeeze()]
        # not_correct_samples.append(  wrong_inputs )
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()
        # fill confusion_matrix
        # for t, p in zip(labels.view(-1), pred_labels.view(-1)):
            # confusion_matrix[t.long(), p.long()] += 1

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    # per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    return accuracy
def get_sfl_loss_n_accuracy(client_model, server_model, criterion, data_loader, args, num_classes=10, sent140_flag=False):
    """ 
    SFL version: Returns accuracy by passing data through client_model 
    then through server_model.
    """
    # 1. Set both models to evaluation mode
    client_model.eval()
    server_model.eval()
    
    total_loss, correctly_labeled_samples = 0, 0
    
    with torch.no_grad(): # Ensure no gradients are computed to save memory/time
        for _, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device=args.device, non_blocking=True), \
                             labels.to(device=args.device, non_blocking=True)

            if sent140_flag:
                labels = torch.ones_like(labels)

            # 2. Sequential SFL Forward Pass
            # Pass through Client
            fx_client = client_model(inputs)
            # Pass result through Server
            outputs = server_model(fx_client)

            # 3. Compute Loss
            if args.data == 'sen140':
                avg_minibatch_loss = criterion(outputs.float(), labels.float())
            else:
                avg_minibatch_loss = criterion(outputs, labels)

            total_loss += avg_minibatch_loss.item() * outputs.shape[0]

            # 4. Compute Accuracy
            if args.data == 'sen140':
                pred_labels = outputs.squeeze() > 0.5
            else:
                _, pred_labels = torch.max(outputs, 1)
            
            pred_labels = pred_labels.view(-1)
            correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()

    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    # Note: If you need avg_loss later, you can return it here too
    return accuracy

def poison_dataset(dataset, args, data_idxs=None, poison_all=False, agent_idx=-1, modify_label=True):
    # if data_idxs != None:
    #     all_idxs = list(set(all_idxs).intersection(data_idxs))
    if data_idxs != None:
        all_idxs = (dataset.targets != args.target_class).nonzero().flatten().tolist()
        all_idxs = list(set(all_idxs).intersection(data_idxs))
    else:
        all_idxs = (dataset.targets != args.target_class).nonzero().flatten().tolist()
    poison_frac = 1 if poison_all else args.poison_frac
    poison_idxs = random.sample(all_idxs, floor(poison_frac * len(all_idxs)))
    for idx in poison_idxs:
        if args.data == 'fedemnist':
            clean_img = dataset.inputs[idx]
        elif args.data in ("tinyimagenet", "ets", "gtsrb"):
            clean_img = dataset[idx][0]
            return
        else:
            clean_img = dataset.data[idx]
        bd_img = add_pattern_bd(clean_img, dataset.targets[idx], args.data, pattern_type=args.pattern_type,
                                agent_idx=agent_idx, attack=args.attack)
        if args.data == 'fedemnist':
            dataset.inputs[idx] = torch.tensor(bd_img)
        elif args.data in ("tinyimagenet", "ets", "gtsrb"):
            # don't do anything for tinyimagenet or ets, we poison it in run time
            return
        else:
            dataset.data[idx] = torch.tensor(bd_img)
        if modify_label:
            dataset.targets[idx] = args.target_class
    return poison_idxs


def init_masks(params, sparsities):
    masks = {}
    for name in params:
        masks[name] = torch.zeros_like(params[name])
        dense_numel = int((1 - sparsities[name]) * torch.numel(masks[name]))
        if dense_numel > 0:
            temp = masks[name].view(-1)
            perm = torch.randperm(len(temp))
            perm = perm[:dense_numel]
            temp[perm] = 1
        masks[name] = masks[name].to("cpu")
    return masks


def vector_to_model(vec, model):
    # Pointer for slicing the vector for each parameter
    state_dict = model.state_dict()
    pointer = 0
    for name in state_dict:
        # The length of the parameter
        num_param = state_dict[name].numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        state_dict[name].data = vec[pointer:pointer + num_param].view_as(state_dict[name]).data
        # Increment the pointer
        pointer += num_param
    model.load_state_dict(state_dict)
    return state_dict

def vector_to_model_wo_load(vec, model):
    # Pointer for slicing the vector for each parameter
    state_dict = model.state_dict()
    pointer = 0
    for name in state_dict:
        # The length of the parameter
        num_param = state_dict[name].numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        state_dict[name].data = vec[pointer:pointer + num_param].view_as(state_dict[name]).data
        # Increment the pointer
        pointer += num_param

    return state_dict


def calculate_sparsities(args, params, tabu=[], distribution="ERK"):
    spasities = {}
    if distribution == "uniform":
        for name in params:
            if name not in tabu:
                spasities[name] = 1 - args.dense_ratio
            else:
                spasities[name] = 0
    elif distribution == "ERK":
        logging.info('initialize by ERK')
        total_params = 0
        for name in params:
            total_params += params[name].numel()
        is_epsilon_valid = False
        # # The following loop will terminate worst case when all masks are in the
        # custom_sparsity_map. This should probably never happen though, since once
        # we have a single variable or more with the same constant, we have a valid
        # epsilon. Note that for each iteration we add at least one variable to the
        # custom_sparsity_map and therefore this while loop should terminate.
        dense_layers = set()

        density = args.dense_ratio
        while not is_epsilon_valid:
            # We will start with all layers and try to find right epsilon. However if
            # any probablity exceeds 1, we will make that layer dense and repeat the
            # process (finding epsilon) with the non-dense layers.
            # We want the total number of connections to be the same. Let say we have
            # for layers with N_1, ..., N_4 parameters each. Let say after some
            # iterations probability of some dense layers (3, 4) exceeded 1 and
            # therefore we added them to the dense_layers set. Those layers will not
            # scale with erdos_renyi, however we need to count them so that target
            # paratemeter count is achieved. See below.
            # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
            #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
            # eps * (p_1 * N_1 + p_2 * N_2) =
            #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
            # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name in params:
                if name in tabu or "running" in name or "track" in name :
                    dense_layers.add(name)
                n_param = np.prod(params[name].shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    raw_probabilities[name] = (
                                                      np.sum(params[name].shape) / np.prod(params[name].shape)
                                              ) ** 1
                    divisor += raw_probabilities[name] * n_param
            epsilon = rhs / divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name in params:
            if name in dense_layers:
                spasities[name] = 0
            else:
                spasities[name] = (1 - epsilon * raw_probabilities[name])
    return spasities


def name_param_to_array(param):
    vec = []
    for name in param:
        # Ensure the parameters are located in the same device
        vec.append(param[name].view(-1))
    return torch.cat(vec)


def vector_to_name_param(vec, name_param_map):
    pointer = 0
    for name in name_param_map:
        # The length of the parameter
        num_param = name_param_map[name].numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        name_param_map[name].data = vec[pointer:pointer + num_param].view_as(name_param_map[name]).data
        # Increment the pointer
        pointer += num_param

    return name_param_map

def to_norm_color(rgb01, dtype, device, IM_MEAN, IM_STD):
    # rgb01: [R,G,B] in [0,1]
    m = torch.tensor(IM_MEAN, dtype=dtype, device=device).view(3,1,1)
    s = torch.tensor(IM_STD, dtype=dtype, device=device).view(3,1,1)
    c = torch.tensor(rgb01, dtype=dtype, device=device).view(3,1,1)
    return (c - m) / s


def add_pattern_bd(x, y, dataset='cifar10', pattern_type='square', agent_idx=-1, attack="DBA"):
    """
    adds a trojan pattern to the image
    """

    # if cifar is selected, we're doing a distributed backdoor attack (i.e., portions of trojan pattern is split between agents, only works for plus)
    if dataset == 'cifar10' or dataset == "cifar100":
        x = np.array(x.squeeze())
        # logging.info(x.shape)
        row = x.shape[0]
        column = x.shape[1]

        if attack == "periodic_trigger":
            for d in range(0, 3):
                for i in range(row):
                    for j in range(column):
                        x[i][j][d] = max(min(x[i][j][d] + 20 * math.sin((2 * math.pi * j * 6) / column), 255), 0)
            # import matplotlib.pyplot as plt
            # plt.imsave("visualization/input_images/backdoor2.png", x)
            # print(y)
            # plt.show()
        else:
            if pattern_type == 'plus':
                start_idx = 5
                size = 6
                if agent_idx == -1:
                    # vertical line
                    # print('xxxxxxx')
                    for d in range(0, 3):
                        for i in range(start_idx, start_idx + size + 1):
                            if d == 2:
                                x[i, start_idx][d] = 0
                            else:
                                x[i, start_idx][d] = 255
                    # horizontal line
                    for d in range(0, 3):
                        for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                            if d == 2:
                                x[start_idx + size // 2, i][d] = 0
                            else:
                                x[start_idx + size // 2, i][d] = 255
                else:
                    if attack == "DBA":
                        # DBA attack
                        # upper part of vertical
                        if agent_idx % 4 == 0:
                            for d in range(0, 3):
                                for i in range(start_idx, start_idx + (size // 2) + 1):
                                    if d == 2:
                                        x[i, start_idx][d] = 0
                                    else:
                                        x[i, start_idx][d] = 255

                        # lower part of vertical
                        elif agent_idx % 4 == 1:
                            for d in range(0, 3):
                                for i in range(start_idx + (size // 2) + 1, start_idx + size + 1):
                                    if d == 2:
                                        x[i, start_idx][d] = 0
                                    else:
                                        x[i, start_idx][d] = 255

                        # left-part of horizontal
                        elif agent_idx % 4 == 2:
                            for d in range(0, 3):
                                for i in range(start_idx - size // 2, start_idx - size // 4 + 1):
                                    if d == 2:
                                        x[start_idx + size // 2, i][d] = 0
                                    else:
                                        x[start_idx + size // 2, i][d] = 255
                        # right-part of horizontal
                        elif agent_idx % 4 == 3:
                            for d in range(0, 3):
                                for i in range(start_idx - size // 4 + 1, start_idx + size // 2 + 1):
                                    if d == 2:
                                        x[start_idx + size // 2, i][d] = 0
                                    else:
                                        x[start_idx + size // 2, i][d] = 255
                    else:
                        # vertical line
                        for d in range(0, 3):
                            for i in range(start_idx, start_idx + size + 1):
                                if d == 2:
                                    x[i, start_idx][d] = 0
                                else:
                                    x[i, start_idx][d] = 255
                        # horizontal line
                        for d in range(0, 3):
                            for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                                if d == 2:
                                    x[start_idx + size // 2, i][d] = 0
                                else:
                                    x[start_idx + size // 2, i][d] = 255

                # import matplotlib.pyplot as plt
                #
                # plt.imsave("visualization/input_images/backdoor2.png", x)
                # print(y)
                # plt.show()

    elif dataset == 'tinyimagenet':
        if pattern_type == 'plus':
            start_idx = 5
            size = 6
            # vertical line
            for d in range(0, 3):
                for i in range(start_idx, start_idx + size + 1):
                    if d == 2:
                        x[d][i][start_idx] = 0
                    else:
                        x[d][i][start_idx] = 1
            # horizontal line
            for d in range(0, 3):
                for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                    if d == 2:
                        x[d][start_idx + size // 2][i] = 0
                    else:
                        x[d][start_idx + size // 2][i] = 1

            # if agent_idx == -1:
            #     # plt.imsave("visualization/input_images/backdoor2.png", x)
            #     print(y)
            #     plt.show()
            # plt.savefig()

    elif dataset == 'fmnist':
        x = np.array(x.squeeze())
        if pattern_type == 'plus':
            start_idx = 5
            size = 6
            if agent_idx == -1:
                # vertical line
                for i in range(start_idx, start_idx + size + 1):
                    x[i, start_idx] = 255
                # horizontal line
                for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                    x[start_idx + size // 2, i] = 255
            else:
                if attack == "DBA":
                    # DBA attack
                    # upper part of vertical
                    if agent_idx % 4 == 0:
                        for i in range(start_idx, start_idx + (size // 2) + 1):
                            x[i, start_idx] = 255

                    # lower part of vertical
                    elif agent_idx % 4 == 1:
                        for i in range(start_idx + (size // 2) + 1, start_idx + size + 1):
                            x[i, start_idx] = 255

                    # left-part of horizontal
                    elif agent_idx % 4 == 2:
                        for i in range(start_idx - size // 2, start_idx - size // 4 + 1):
                            x[start_idx + size // 2, i] = 255

                    # right-part of horizontal
                    elif agent_idx % 4 == 3:
                        for i in range(start_idx - size // 4 + 1, start_idx + size // 2 + 1):
                            x[start_idx + size // 2, i] = 255
                else:
                    # vertical line
                    for i in range(start_idx, start_idx + size + 1):
                        x[i, start_idx] = 255
                    # horizontal line
                    for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                        x[start_idx + size // 2, i] = 255

    elif dataset == 'gtsrb':
        IM_MEAN = [0.485, 0.456, 0.406]
        IM_STD = [0.229, 0.224, 0.225]
        # Case A: torch Tensor CHW
        if isinstance(x, torch.Tensor):
            # Assume CHW float; treat as normalized to ImageNet stats if you used those transforms.
            C, H, W = x.shape
            normalized = True  # set False if you normalize with mean/std=0.5
            if attack == "periodic_trigger":
                # Light periodic modulation along width; apply on each channel
                jj = torch.arange(W, device=x.device, dtype=x.dtype).view(1, 1, W)
                delta = 20.0/255.0 * torch.sin(2 * math.pi * jj * 6 / max(1, W))  # in [0,1] scale
                if normalized:
                    # Map delta to normalized scale approximately by dividing by std
                    std = torch.tensor(IM_STD, dtype=x.dtype, device=x.device).view(3,1,1)
                    x = x + delta / std
                else:
                    x = x + delta
                x = x.clamp(-10, 10)  # keep bounded
                return x

            if pattern_type == 'square':
                s = max(2, W // 16)
                # white square (1,1,1) in normalized space if needed
                color = torch.ones(3,1,1, dtype=x.dtype, device=x.device)
                if normalized:
                    m = torch.tensor(IM_MEAN, dtype=x.dtype, device=x.device).view(3,1,1)
                    sdt = torch.tensor(IM_STD, dtype=x.dtype, device=x.device).view(3,1,1)
                    color = (color - m) / sdt
                x[:, H - s:H, W - s:W] = color
                return x
            else:
                # plus trigger near bottom-right, DBA optional via agent_idx % 4
                half = max(1, W // 32)
                ci, cj = H - (3 + half), W - (3 + half)
                # yellow-ish (R=1,G=1,B=0)
                color = [1.0, 1.0, 0.0]
                if normalized:
                    color_t = to_norm_color(color, x.dtype, x.device, IM_MEAN, IM_STD)
                else:
                    color_t = torch.tensor(color, dtype=x.dtype, device=x.device).view(3,1,1)

                def set_pix(i, j):
                    if 0 <= i < H and 0 <= j < W:
                        x[:, i, j] = color_t.view(3)

                if attack == "DBA" and agent_idx is not None and agent_idx >= 0:
                    part = agent_idx % 4
                    if part == 0:
                        # upper vertical
                        for i in range(ci - half, ci + 1):
                            set_pix(i, cj)
                    elif part == 1:
                        # lower vertical
                        for i in range(ci + 1, ci + half + 1):
                            set_pix(i, cj)
                    elif part == 2:
                        # left horizontal
                        for j in range(cj - half, cj + 1):
                            set_pix(ci, j)
                    else:
                        # right horizontal
                        for j in range(cj + 1, cj + half + 1):
                            set_pix(ci, j)
                else:
                    # full plus
                    for i in range(ci - half, ci + half + 1):
                        set_pix(i, cj)
                    for j in range(cj - half, cj + half + 1):
                        set_pix(ci, j)
                return x

        # Case B: PIL or numpy HWC
        x = np.array(x)
        if x.ndim == 2:
            x = np.stack([x]*3, axis=-1)
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        H, W = x.shape[:2]

        if attack == "periodic_trigger":
            for d in range(3):
                for i in range(H):
                    for j in range(W):
                        x[i, j, d] = np.clip(
                            x[i, j, d] + 20 * math.sin((2 * math.pi * j * 6) / max(1, W)), 0, 255
                        )
        else:
            if pattern_type == 'square':
                s = max(2, W // 16)
                x[H - s:H, W - s:W, :] = 255
            else:  # 'plus'
                half = max(1, W // 32)
                ci, cj = H - (3 + half), W - (3 + half)

                def set_pix_np(i, j):
                    if 0 <= i < H and 0 <= j < W:
                        x[i, j, 0] = 255
                        x[i, j, 1] = 255
                        x[i, j, 2] = 0

                if attack == "DBA" and agent_idx is not None and agent_idx >= 0:
                    part = agent_idx % 4
                    if part == 0:
                        for i in range(ci - half, ci + 1):
                            set_pix_np(i, cj)
                    elif part == 1:
                        for i in range(ci + 1, ci + half + 1):
                            set_pix_np(i, cj)
                    elif part == 2:
                        for j in range(cj - half, cj + 1):
                            set_pix_np(ci, j)
                    else:
                        for j in range(cj + 1, cj + half + 1):
                            set_pix_np(ci, j)
                else:
                    for i in range(ci - half, ci + half + 1):
                        set_pix_np(i, cj)
                    for j in range(cj - half, cj + half + 1):
                        set_pix_np(ci, j)
        return x

    elif dataset == 'ets':
        # Accept PIL, numpy HWC uint8, or torch Tensor (CHW in [0,1] or normalized).
        orig_type = type(x)
        is_torch = False
        is_normalized = False

        # Convert to numpy HWC uint8 for drawing
        if isinstance(x, torch.Tensor):
            is_torch = True
            # Detect normalization by typical ImageNet std magnitude
            if x.dtype.is_floating_point and x.abs().max().item() > 1.5:
                # very likely normalized already; we still can place a binary pattern that will be re-normalized later
                is_normalized = True
            # Bring to CPU numpy
            if x.ndim != 3 or x.shape[0] not in (1, 3):
                raise ValueError(f"Unexpected tensor shape for ETS: {tuple(x.shape)}")
            # CHW -> HWC
            x_np = x.detach().cpu().numpy()
            if x_np.dtype != np.uint8:
                # assume float in [0,1] or normalized float; clamp to [0,1] then scale
                x_np = np.clip(x_np, 0.0, 1.0)
                x_np = (x_np * 255.0).round().astype(np.uint8)
            x_np = np.transpose(x_np, (1, 2, 0))
        else:
            x_np = np.array(x)
            if x_np.ndim == 3 and x_np.shape[0] in (1, 3) and x_np.shape[2] not in (1, 3):
                # looks like CHW -> convert to HWC
                x_np = np.transpose(x_np, (1, 2, 0))
            if x_np.dtype != np.uint8:
                x_np = np.clip(x_np, 0, 255).astype(np.uint8)

        H, W = x_np.shape[:2]
        # Ensure we have 3 channels
        if x_np.ndim == 2:
            x_np = np.stack([x_np]*3, axis=-1)
        elif x_np.shape[2] == 1:
            x_np = np.repeat(x_np, 3, axis=2)

        # Draw trigger
        if attack == "periodic_trigger":
            for d in range(3):
                for i in range(H):
                    for j in range(W):
                        x_np[i, j, d] = np.clip(
                            x_np[i, j, d] + 20 * math.sin((2 * math.pi * j * 6) / W),
                            0, 255
                        )
        else:
            if pattern_type == 'square':
                # 3x3 square at bottom-right (correct H/W order)
                x_np[max(0, H-3):H, max(0, W-3):W, :] = 255
            else:
                # small plus near bottom-right
                ci, cj, half = max(2, H-5), max(2, W-5), 2
                # vertical line
                for i in range(ci - half, ci + half + 1):
                    if 0 <= i < H and 0 <= cj < W:
                        # yellow-ish (R,G high, B low) like your CIFAR version
                        x_np[i, cj, 0] = 255
                        x_np[i, cj, 1] = 255
                        x_np[i, cj, 2] = 0
                # horizontal line
                for j in range(cj - half, cj + half + 1):
                    if 0 <= ci < H and 0 <= j < W:
                        x_np[ci, j, 0] = 255
                        x_np[ci, j, 1] = 255
                        x_np[ci, j, 2] = 0

        # Convert back to original type/layout
        if is_torch:
            out = torch.from_numpy(x_np.astype(np.uint8)).permute(2, 0, 1)  # HWC->CHW
            # If the original tensor was float, map back to float [0,1]; normalization should happen later in transform
            out = out.float() / 255.0
            return out
        else:
            return x_np

    elif dataset == 'mnist':
        x = np.array(x.squeeze())
        if pattern_type == 'plus':
            start_idx = 1
            size = 2
            if agent_idx == -1:
                # vertical line
                for i in range(start_idx, start_idx + size + 1):
                    x[i, start_idx] = 255
                # horizontal line
                for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                    x[start_idx + size // 2, i] = 255
            else:
                if attack == "DBA":
                    # DBA attack
                    # upper part of vertical
                    if agent_idx % 4 == 0:
                        for i in range(start_idx, start_idx + (size // 2) + 1):
                            x[i, start_idx] = 255

                    # lower part of vertical
                    elif agent_idx % 4 == 1:
                        for i in range(start_idx + (size // 2) + 1, start_idx + size + 1):
                            x[i, start_idx] = 255

                    # left-part of horizontal
                    elif agent_idx % 4 == 2:
                        for i in range(start_idx - size // 2, start_idx - size // 4 + 1):
                            x[start_idx + size // 2, i] = 255

                    # right-part of horizontal
                    elif agent_idx % 4 == 3:
                        for i in range(start_idx - size // 4 + 1, start_idx + size // 2 + 1):
                            x[start_idx + size // 2, i] = 255
                else:
                    # vertical line
                    for i in range(start_idx, start_idx + size + 1):
                        x[i, start_idx] = 255
                    # horizontal line
                    for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                        x[start_idx + size // 2, i] = 255
    # import matplotlib.pyplot as plt
    # if agent_idx == -1:
    #     # plt.imsave("visualization/input_images/backdoor2.png", x)
    #     plt.imshow(x)
    #     print(y)
    #     plt.show()
    return x

def extract_last_layer(net, model=""):
    bias, weight = None, None

    weights = list(net.values())

    weight = weights[-2].cpu().numpy()
    bias = weights[-1].cpu().numpy()

    return bias, weight 

def vector_to_net_dict(vec: torch.Tensor, net_dict) -> None:
    r"""Convert one vector to the net parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    net_dict = net_dict.state_dict()
    # net_dict = {k: net_dict[k] for k in net_dict.keys()}
    pointer = 0
    for param in net_dict.values():
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
    return net_dict


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        # if key.split('.')[-1] == 'num_batches_tracked':
        #     continue
        vec.append(param.view(-1))
    return torch.cat(vec)



# def setup_logging(args):
#     """
#     Sets up the logging environment and creates necessary directories.
    
#     Args:
#         args: Arguments object containing logging parameters like non_iid, alpha, data, and aggr.
        
#     Returns:
#         dir_path: The directory path where logs and backup files are stored.
#     """
#     log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
#     root_logger = logging.getLogger()
#     root_logger.setLevel(logging.INFO)
    
#     # Clear existing handlers to avoid duplicates
#     if root_logger.hasHandlers():
#         root_logger.handlers.clear()
    
#     logPath = "logs"
#     time_str = time.strftime("%Y-%m-%d-%H-%M")

#     if args.non_iid:
#         iid_str = 'noniid(%.1f)' % args.beta
#     else:
#         iid_str = 'iid'

#     args.exp_name = iid_str + '_pr(%.1f)' % args.poison_frac

#     if args.exp_name_extra != '':
#         args.exp_name += '_%s' % args.exp_name_extra

#     fileName = "%s_%s" % (time_str, args.exp_name)

#     dir_path = '%s/%s/attack_%s_ar_%.2f/defense_%s/%s/' % (logPath, args.data, args.attack, args.num_corrupt / args.num_agents, args.aggr, fileName)
#     file_path = dir_path + 'backup_file/'

#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)

#     if not os.path.exists(file_path):
#         os.makedirs(file_path)

#     backup_file = ['aggregation.py', 'federated.py', 'agent.py']

#     for file in backup_file:
#         copyfile('./%s' % file, file_path + file)
    
#     # Set up file handler for logging
#     file_handler = logging.FileHandler(os.path.join(dir_path, f"{fileName}.log"))
#     file_handler.setFormatter(log_formatter)
#     root_logger.addHandler(file_handler)
    
#     # Set up console handler for logging
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.INFO)
#     console_handler.setFormatter(log_formatter)
#     root_logger.addHandler(console_handler)
    
#     # Log initial arguments
#     logging.info(args)
#     return dir_path
