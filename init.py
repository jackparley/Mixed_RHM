import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import datasets
import models
import measures


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]  # Your dataset returns (x, y)
        return data, label, idx


class CosineWarmupLR(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def init_data_mixed(args):
    """
    Initialise dataset.

    Returns:
        Two dataloaders for train and test set.
    """
    if args.dataset == "mixed_rhm":

        #test_size = min(args.max_data - args.train_size, 20000)
        test_size=20000
        dataset = datasets.MixedRandomHierarchyModel(
            num_features=args.num_features,  # vocabulary size
            num_classes=args.num_classes,  # number of classes
            fraction_rules=args.fraction_rules,  # number of synonymic low-level representations (multiplicity)
            rule_sequence_type=args.rule_sequence_type,  # type of rule sequence
            s_2=2,
            s_3=3,  # size of the low-level representations
            num_layers=args.num_layers,
            max_data=args.max_data,  # number of levels in the hierarchy
            seed_rules=args.seed_rules,
            seed_sample=args.seed_sample,
            train_size=args.train_size,
            test_size=test_size,
            input_format=args.input_format,
            whitening=args.whitening,
            padding=args.padding,
            padding_central=args.padding_central,
            padding_tail=args.padding_tail,
            replacement=args.replacement,
            d_5_set=args.d_5_set,
            non_overlapping=args.non_overlapping,
            check_overlap=args.check_overlap,
        )
    elif args.dataset == "mixed_rhm_varying_tree":
        test_size = 20000
        dataset = datasets.MixedRandomHierarchyModel_varying_tree(
            num_features=args.num_features,  # vocabulary size
            num_classes=args.num_classes,  # number of classes
            fraction_rules=args.fraction_rules,  # number of synonymic low-level representations (multiplicity)
            s_2=2,
            s_3=3,  # size of the low-level representations
            num_layers=args.num_layers,
            seed_rules=args.seed_rules,
            seed_sample=args.seed_sample,
            train_size=args.train_size,
            test_size=test_size,
            padding_tail=args.padding_tail,
            padding_central=args.padding_central,
            padding_classification=args.padding_classification,
            return_type=args.return_type,
            cover_all=args.cover_all,
            d_5_4_set=args.d_5_4_set,
            eta_set=args.eta_set,
            eta=args.eta,
            top_ter=args.top_ter,
            non_overlapping=args.non_overlapping,
            return_topology=args.return_topology,
            input_format=args.input_format,
            whitening=args.whitening,
        )
        print("Dataset loaded")
    rules_rhm = dataset.rules
    trainset = torch.utils.data.Subset(dataset, range(args.train_size))
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    testset = torch.utils.data.Subset(
        dataset, range(args.train_size, args.train_size + test_size)
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=False, num_workers=0
    )
    indexed_testset = IndexedDataset(testset)
    test_loader_indexed = torch.utils.data.DataLoader(
        indexed_testset, batch_size=len(testset), shuffle=False
    )
    if args.return_type == 1:
        tree_ints = dataset.tree_types
        subset_indices = range(args.train_size, args.train_size + test_size)
        subset_data_type = [tree_ints[i] for i in subset_indices]
        return train_loader, test_loader_indexed, subset_data_type
    elif args.check_overlap==1:
        tree_ints = dataset.overlap_flags
        subset_indices = range(args.train_size, args.train_size + test_size)
        subset_data_type = [tree_ints[i] for i in subset_indices]
        return train_loader, test_loader_indexed, subset_data_type
    else:
        return train_loader, test_loader, rules_rhm


def init_model_mixed(args):
    """
    Initialise machine-learning model.
    """
    torch.manual_seed(args.seed_model)

    if args.model == "hcnn_mixed":
        model = models.hCNN_mixed(
            rule_sequence_type=args.rule_sequence_type,
            in_channels=args.num_features,
            nn_dim=args.width,
            out_channels=args.num_classes,
            num_layers=args.depth,
            bias=args.bias,
            norm="mf",  # TODO: add arg for different norm
        )
        args.lr *= args.width  # TODO: modify for different norm
    elif args.model == "hcnn_Gen":
        model = models.hCNN_Gen(
            in_channels=args.num_features,
            nn_dim=args.width,
            out_channels=args.num_classes,
            num_layers=args.depth,
            final_dim=args.final_dim,
            bias=args.bias,
            norm="mf",  # TODO: add arg for different norm
        )
        args.lr *= args.width  # TODO: modify for different norm
    elif args.model == "hcnn_Gen_top_fix":
        model = models.hCNN_Gen_top_fix(
            in_channels=args.num_features,
            nn_dim=args.width,
            out_channels=args.num_classes,
            num_layers=args.depth,
            final_dim=args.final_dim,
            bias=args.bias,
            norm="mf",  # TODO: add arg for different norm
        )
        args.lr *= args.width  # TODO: modify for different norm
    elif args.model == "hcnn_no_sharing_Gen":
        model = models.hCNN_no_sharing_Gen(
            in_channels=args.num_features,
            nn_dim=args.width,
            out_channels=args.num_classes,
            bias=args.bias,
            norm="mf",  # TODO: add arg for different norm
        )
        args.lr *= args.width  # TODO: modify for different norm
    elif args.model == "hcnn_Gen_MLP":
        model = models.hCNN_Gen_MLP(
            in_channels=args.num_features,
            nn_dim=args.width,
            out_channels=args.num_classes,
            num_layers=args.depth,
            final_dim=args.final_dim,
            mlp_dim=args.mlp_dim,
            bias=args.bias,
            norm="mf",  # TODO: add arg for different norm
        )
        args.lr *= args.width  # TODO: modify for different norm
    elif args.model == "hcnn_sharing":
        model = models.hCNN_sharing(
            in_channels=args.num_features,
            nn_dim=args.width,
            nn_dim_2=args.width_2,
            out_channels=args.num_classes,
            bias=args.bias,
            norm="mf",  # TODO: add arg for different norm
        )
        args.lr *= args.width_2  # TODO: modify for different norm
    elif args.model == "hcnn_no_sharing":
        model = models.hCNN_no_sharing(
            in_channels=args.num_features,
            nn_dim=args.width,
            nn_dim_2=args.width_2,
            out_channels=args.num_classes,
            bias=args.bias,
            norm="mf",  # TODO: add arg for different norm
        )
        args.lr *= args.width_2  # TODO: modify for different norm
    elif args.model == "hcnn_inside":
        model = models.hCNN_inside(
            in_channels=args.num_features,
            nn_dim=args.width,
            out_channels=args.num_classes,
            bias=args.bias,
            norm="mf",  # TODO: add arg for different norm
        )
        args.lr *= args.width  # TODO: modify for different norm
    elif args.model == "hcnn_inside_L_2":
        model = models.hCNN_inside_L_2(
            in_channels=args.num_features,
            nn_dim=args.width,
            out_channels=args.num_classes,
            bias=args.bias,
            norm="mf",  # TODO: add arg for different norm
        )
        args.lr *= args.width  # TODO: modify for different norm
    elif args.model == "hcnn_inside_L_2_tree_topologies":
        model = models.hCNN_inside_L_2_tree_topologies(
            in_channels=args.num_features,
            nn_dim=args.width,
            out_channels=args.num_classes,
            bias=args.bias,
            norm="mf",  # TODO: add arg for different norm
        )
        args.lr *= args.width  # TODO: modify for different norm
    
    elif args.model == "hcnn_inside_L_3":
        model = models.hCNN_inside_L_3(
            in_channels=args.num_features,
            nn_dim=args.width,
            out_channels=args.num_classes,
            bias=args.bias,
            norm="mf",  # TODO: add arg for different norm
        )
        args.lr *= args.width  # TODO: modify for different norm

    elif args.model == "hcnn_inside_L_4":
        model = models.hCNN_inside_L_4(
            in_channels=args.num_features,
            nn_dim=args.width,
            out_channels=args.num_classes,
            bias=args.bias,
            norm="mf",  # TODO: add arg for different norm
        )
        args.lr *= args.width  # TODO: modify for different norm

    elif args.model == "fcn":

        if args.depth == 0:
            model = models.Perceptron(
                input_dim=args.num_tokens * args.num_features,
                out_dim=args.num_classes,
                norm=args.num_tokens**0.5,
            )
        else:

            assert args.width is not None, "FCN model requires argument width!"
            model = models.MLP(
                input_dim=args.num_tokens * args.num_features,
                nn_dim=args.width,
                out_dim=args.num_classes,
                num_layers=args.depth,
                bias=args.bias,
                norm="mf",  # TODO: add arg for different norm
            )
            args.lr *= args.width  # TODO: modify for different norm
    elif "transformer" in args.model:

        assert (
            args.num_heads is not None
        ), "transformer model requires argument num_heads!"
        assert (
            args.embedding_dim is not None
        ), "transformer model requires argument embedding_dim!"

        if args.model == "transformer_mla":

            model = models.MLA(
                vocab_size=args.num_features+1,  # +1 for classification token
                embedding_dim=args.embedding_dim,
                num_heads=args.num_heads,
                num_layers=args.depth,
            )

    else:
        raise ValueError("model argument is invalid!")

    model = model.to(args.device)

    return model


def init_data(args):
    """
    Initialise dataset.

    Returns:
        Two dataloaders for train and test set.
    """
    if args.dataset == "rhm":

        probability = None

        if args.zipf is not None:
            assert args.layer is not None, "zipf law requires layer of application"
            probability = {}
            for l in range(args.num_layers):
                probability[l] = torch.ones(args.num_synonyms) / args.num_synonyms
            zipf_prob = torch.zeros(args.num_synonyms)
            if args.zipf == "inf":
                zipf_prob[0] = 1.0
                probability[args.layer - 1] = zipf_prob
            else:
                for i in range(args.num_synonyms):
                    zipf_prob[i] = (i + 1) ** (-1 - float(args.zipf))
                probability[args.layer - 1] = zipf_prob / zipf_prob.sum()

        dataset = datasets.RandomHierarchyModel(
            num_features=args.num_features,  # vocabulary size
            num_synonyms=args.num_synonyms,  # features multiplicity
            num_layers=args.num_layers,  # number of layers
            num_classes=args.num_classes,  # number of classes
            tuple_size=args.tuple_size,  # number of branches of the tree
            probability=probability,
            seed_rules=args.seed_rules,
            train_size=args.train_size,
            test_size=args.test_size,
            seed_sample=args.seed_sample,
            input_format=args.input_format,
            whitening=args.whitening,  # 1 for standardising input
            replacement=args.replacement,  # Automatically true for num_data > 1e19
            bonus=args.bonus,  # bonus dictionary
        )

        args.input_size = args.tuple_size**args.num_layers
        if args.num_tokens < args.input_size:  # only take last num_tokens positions
            dataset.features = dataset.features[:, :, -args.num_tokens :]

    else:
        raise ValueError("dataset argument is invalid!")

    if args.mode == "masked":  # hide last feature from input and set it as label

        dataset.labels = torch.argmax(dataset.features[:, :, -1], dim=1)

        if "fcn" in args.model:  # for fcn remove masked token from the input
            dataset.features = dataset.features[:, :, :-1]
            args.num_tokens -= 1
            if args.bonus:
                if "synonyms" in args.bonus:
                    for k in args.bonus["synonyms"].keys():
                        args.bonus["synonyms"][k] = args.bonus["synonyms"][k][:, :, :-1]
                if "noise" in args.bonus:
                    for k in args.bonus["noise"].keys():
                        args.bonus["noise"][k] = args.bonus["noise"][k][:, :, :-1]

        else:  # for other models replace masked token with ones
            mask = torch.ones(args.num_features) * args.num_features**-0.5
            mask = torch.tile(mask, [args.train_size + args.test_size, 1])
            dataset.features[:, :, -1] = mask
            if args.bonus:
                if "synonyms" in args.bonus:
                    for k in args.bonus["synonyms"].keys():
                        args.bonus["synonyms"][k][:, :, -1] = mask[
                            -args.bonus["size"] :
                        ]
                if "noise" in args.bonus:
                    for k in args.bonus["noise"].keys():
                        args.bonus["noise"][k][:, :, -1] = mask[-args.bonus["size"] :]

    if "fcn" in args.model:  # fcn requires flattening of the input
        dataset.features = dataset.features.transpose(1, 2).flatten(
            start_dim=1
        )  # groups of adjacent num_features correspond to a pixel
        if args.bonus:
            if "synonyms" in args.bonus:
                for k in args.bonus["synonyms"].keys():
                    args.bonus["synonyms"][k] = (
                        args.bonus["synonyms"][k].transpose(1, 2).flatten(start_dim=1)
                    )
            if "noise" in args.bonus:
                for k in args.bonus["noise"].keys():
                    args.bonus["noise"][k] = (
                        args.bonus["noise"][k].transpose(1, 2).flatten(start_dim=1)
                    )

    if (
        "transformer" in args.model
    ):  # transformer requires [batch_size, seq_len, num_channels] format
        dataset.features = dataset.features.transpose(1, 2)
        if args.bonus:
            if "synonyms" in args.bonus:
                for k in args.bonus["synonyms"].keys():
                    args.bonus["synonyms"][k] = args.bonus["synonyms"][k].transpose(
                        1, 2
                    )
            if "noise" in args.bonus:
                for k in args.bonus["noise"].keys():
                    args.bonus["noise"][k] = args.bonus["noise"][k].transpose(1, 2)

        # TODO: append classification token to input for transformers used in class

    if args.bonus:
        if "rules" in args.bonus:
            args.bonus["rules"] = dataset.rules
        if "synonyms" in args.bonus:
            for k in args.bonus["synonyms"].keys():
                args.bonus["synonyms"][k] = args.bonus["synonyms"][k].to(args.device)
        if "noise" in args.bonus:
            for k in args.bonus["noise"].keys():
                args.bonus["noise"][k] = args.bonus["noise"][k].to(args.device)

    if args.bonus:
        args.bonus["features"] = dataset.features[-args.bonus["size"] :]
        args.bonus["labels"] = dataset.labels[-args.bonus["size"] :]

    trainset = torch.utils.data.Subset(dataset, range(args.train_size))
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    if args.test_size:
        testset = torch.utils.data.Subset(
            dataset, range(args.train_size, args.train_size + args.test_size)
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=1024, shuffle=False, num_workers=0
        )
    else:
        test_loader = None

    return train_loader, test_loader


def init_model(args):
    """
    Initialise machine-learning model.
    """
    torch.manual_seed(args.seed_model)

    if args.model == "fcn":

        if args.depth == 0:
            model = models.Perceptron(
                input_dim=args.num_tokens * args.num_features,
                out_dim=args.num_classes,
                norm=args.num_tokens**0.5,
            )
        else:

            assert args.width is not None, "FCN model requires argument width!"
            model = models.MLP(
                input_dim=args.num_tokens * args.num_features,
                nn_dim=args.width,
                out_dim=args.num_classes,
                num_layers=args.depth,
                bias=args.bias,
                norm="mf",  # TODO: add arg for different norm
            )
            args.lr *= args.width  # TODO: modify for different norm

    elif args.model == "hcnn":

        assert args.width is not None, "CNN model requires argument width!"
        assert args.filter_size is not None, "CNN model requires argument filter_size!"
        exponent = math.log(args.num_tokens) / math.log(args.filter_size)
        assert (
            args.depth == exponent
        ), "hierarchical CNN requires num_tokens == filter_size**depth"

        model = models.hCNN(
            input_dim=args.num_tokens,
            patch_size=args.filter_size,
            in_channels=args.num_features,
            nn_dim=args.width,
            out_channels=args.num_classes,
            num_layers=args.depth,
            bias=args.bias,
            norm="mf",  # TODO: add arg for different norm
        )
        args.lr *= args.width  # TODO: modify for different norm

    elif args.model == "hlcn":

        assert args.width is not None, "LCN model requires argument width!"
        assert args.filter_size is not None, "LCN model requires argument filter_size!"
        exponent = math.log(args.num_tokens) / math.log(args.filter_size)
        assert (
            args.depth == exponent
        ), "hierarchical LCN requires num_tokens == filter_size**depth"

        model = models.hLCN(
            input_dim=args.num_tokens,
            patch_size=args.filter_size,
            in_channels=args.num_features,
            nn_dim=args.width,
            out_channels=args.num_classes,
            num_layers=args.depth,
            bias=args.bias,
            norm="mf",  # TODO: add arg for different norm
        )
        args.lr *= args.width  # TODO: modify for different norm

    elif "transformer" in args.model:

        assert (
            args.num_heads is not None
        ), "transformer model requires argument num_heads!"
        assert (
            args.embedding_dim is not None
        ), "transformer model requires argument embedding_dim!"

        if args.model == "transformer_mla":

            model = models.MLA(
                vocab_size=args.num_features,
                block_size=args.num_tokens,
                embedding_dim=args.embedding_dim,
                num_heads=args.num_heads,
                num_layers=args.depth,
            )

    else:
        raise ValueError("model argument is invalid!")

    model = model.to(args.device)

    return model


def init_training(model, args):
    """
    Initialise training algorithm.
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optim == "sgd_layerwise":
        optimizer = optim.SGD(
            [
                {"params": model.conv1.parameters(), "lr": args.lr * args.width_2},
                {"params": model.conv2.parameters(), "lr": args.lr * args.width_2},
                {
                    "params": [model.readout],
                    "lr": args.lr * args.width_2,
                },  # readout also scaled by final width
            ],
            momentum=args.momentum,
        )
    elif args.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError("optimizer is invalid (sgd, adam)!")

    if args.scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.max_iters)
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.scheduler_time, eta_min=0.1 * args.lr
        )
    elif args.scheduler == "warmup":
        scheduler = CosineWarmupLR(
            optimizer, args.scheduler_time, max_iters=args.max_iters
        )

    return criterion, optimizer, scheduler


def init_output_mixed(model, criterion, train_loader, test_loader, args):
    """
    Initialise output of the experiment.

    Returns:
        list with the dynamics, best model.
    """

    trainloss, trainacc = measures.test(model, train_loader, args.device)
    # testloss, testacc = measures.test(model, test_loader, args.device)
    testloss = trainloss
    testacc = trainacc
    print_dict = {
        "t": 0,
        "trainloss": trainloss,
        "trainacc": trainacc,
        "testloss": testloss,
        "testacc": testacc,
    }
    dynamics = [print_dict]

    best = {"step": 0, "model": None, "loss": testloss}

    return dynamics, best


def init_output(model, criterion, train_loader, test_loader, args):
    """
    Initialise output of the experiment.

    Returns:
        list with the dynamics, best model.
    """

    trainloss, trainacc = measures.test(model, train_loader, args.device)
    testloss, testacc = measures.test(model, test_loader, args.device)

    print_dict = {
        "t": 0,
        "trainloss": trainloss,
        "trainacc": trainacc,
        "testloss": testloss,
        "testacc": testacc,
    }
    if args.bonus:
        if "synonyms" in args.bonus:
            print_dict["synonyms"] = measures.sensitivity(
                model, args.bonus["features"], args.bonus["synonyms"], args.device
            )
        if "noise" in args.bonus:
            print_dict["noise"] = measures.sensitivity(
                model, args.bonus["features"], args.bonus["noise"], args.device
            )
    dynamics = [print_dict]

    best = {"step": 0, "model": None, "loss": testloss}

    return dynamics, best



def log2ckpt(end, freq):
    current = 1.0
    factor = 2 ** (1.0 / freq)
    threshold = 2 ** (math.ceil(math.log(1.0 / (factor - 1))) + 1)
    checkpoints = []

    while current < threshold:
        checkpoints.append(round(current))
        current += 1

    while round(current) < end:
        checkpoints.append(round(current))
        current *= factor
 
    checkpoints.append(round(end))

    return checkpoints


def init_loglinckpt(step, end, freq, log_linear_switch):
    """
    Initialise checkpoint iterators.

    Args:
        step: linear step size
        end: maximum value
        freq: frequency of checkpoints in logscale
        log_linear_switch: value (in steps) after which to switch from log to linear growth
    Returns:
        Two iterators: linear and mixed (logarithmic first, then linear after switch).
    """
    # Linear checkpoints (same as before)
    factor = 2 ** (1.0 / freq)
    multiplier = 2 ** (math.ceil(math.log(1.0 / (factor - 1))) + 1)

    lin_ckpts = log2ckpt(multiplier * step, freq)
    current = lin_ckpts[-1] + step
    while current <= end:
        lin_ckpts.append(current)
        current += step
    lin_ckpts.append(0)

    # Logarithmic until log_linear_switch
    log_ckpts = []
    current = 1.0
    prev = None
    while current * step < log_linear_switch:
        prev = round(current) * step
        log_ckpts.append(prev)
        current *= factor

    # After reaching log_linear_switch:
    if prev is None:
        raise ValueError("log_linear_switch too small, no logarithmic steps generated.")

    # Calculate the last delta seen in log space
    next_log = round(current) * step
    last_log_delta = next_log - prev
    if last_log_delta <= 0:
        last_log_delta = step  # fallback

    # Now continue linearly with that last_log_delta
    current = prev + last_log_delta
    while current <= end:
        log_ckpts.append(current)
        current += last_log_delta
    log_ckpts.append(0)

    return iter(lin_ckpts), iter(log_ckpts)
