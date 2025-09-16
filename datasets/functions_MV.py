
from itertools import *
import warnings
import copy
import sys

import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import defaultdict

import os
import argparse
import pickle



import torch
from torch import nn

from torch.nn import functional as F
from sklearn.cluster import KMeans



def sample_non_overlapping_mixed_rules(v, n, m_2, m_3, s_2, s_3, L, seed):
    random.seed(seed)

    def tensor_default():
        return torch.empty(0)

    tuples_2 = list(product(*[range(v) for _ in range(s_2)]))
    tuples_3 = list(product(*[range(v) for _ in range(s_3)]))

    rules = defaultdict(lambda: defaultdict(tensor_default))
    num_symbols=v
    for l in range(L):
        # Determine how many nonterminals at this level
        

        # Sample binary rules
        binary_tuples = random.sample(tuples_2, num_symbols * m_2)
        binary_rule_set = set(binary_tuples)
        binary_rules = torch.tensor(binary_tuples).reshape(num_symbols, m_2, s_2)

        # Rejection sampling for ternary rules without overlapping binary pairs
        valid_ternary = []
        used_ternary = set()
        for t in random.sample(tuples_3, len(tuples_3)):  # shuffled
            if len(valid_ternary) >= num_symbols * m_3:
                break
            (a, b, c) = t
            if (a, b) in binary_rule_set or (b, c) in binary_rule_set:
                continue
            if t in used_ternary:
                continue
            valid_ternary.append(t)
            used_ternary.add(t)

        if len(valid_ternary) < num_symbols * m_3:
            raise ValueError(f"Not enough valid ternary rules found at layer {l}.")

        ternary_rules = torch.tensor(valid_ternary).reshape(num_symbols, m_3, s_3)

        rules[l][0] = binary_rules
        rules[l][1] = ternary_rules

    return rules

def sample_mixed_rules(v, n, m_2, m_3, s_2, s_3, L, seed):
    random.seed(seed)

    # Define a callable for the inner defaultdict to return empty tensors
    def tensor_default():
        return torch.empty(0)

    tuples_2 = list(product(*[range(v) for _ in range(s_2)]))
    tuples_3 = list(product(*[range(v) for _ in range(s_3)]))

    # Define the nested defaultdict structure
    rules = defaultdict(lambda: defaultdict(tensor_default))

    # Initialize the grammar with sampled tensors
    rules[0][0] = torch.tensor(random.sample(tuples_2, n * m_2)).reshape(n, m_2, s_2)
    rules[0][1] = torch.tensor(random.sample(tuples_3, n * m_3)).reshape(n, m_3, s_3)
    for i in range(1, L):
        rules[i][0] = torch.tensor(random.sample(tuples_2, v * m_2)).reshape(
            v, m_2, s_2
        )
        rules[i][1] = torch.tensor(random.sample(tuples_3, v * m_3)).reshape(
            v, m_3, s_3
        )

    return rules

def index_to_choice_d_5(index, n,m2,m3):
    Pmax=n*m2*m3*m2
    if index <Pmax:

        bases = [n, m2,m3,m2]  # Alternating base sizes
        #index -= 1  # Convert to 0-based index
        choice = []

        # Compute the total number of possibilities
        total_combinations = 1
        for base in bases:
            total_combinations *= base

        # Extract choices one by one
        for base in bases:
            total_combinations //= base  # Reduce divisor dynamically
            choice.append(index // total_combinations + 1)
            index %= total_combinations  # Reduce index to the remainder
    else:
        index = index-Pmax
        bases = [n, m2,m2,m3]
        choice = []

        # Compute the total number of possibilities
        total_combinations = 1
        for base in bases:
            total_combinations *= base

        # Extract choices one by one
        for base in bases:
            total_combinations //= base  # Reduce divisor dynamically
            choice.append(index // total_combinations + 1)
            index %= total_combinations  # Reduce index to the remainder
    return choice

def sample_data_from_indices_d_5(
    samples, rules, n, m_2, m_3
):
    L = len(rules)
    Pmax=n*m_2*m_3*m_2
    all_features = []
    labels = []
    #samples = samples + 1
    for sample in samples:
        chosen_rules = index_to_choice_d_5(sample, n, m_2, m_3)
        labels.append(chosen_rules[0] - 1)
        # print(chosen_rules[0])
        chosen_rules = [x - 1 for x in chosen_rules]
        # for label in labels:
        # Initialize the current symbols with the start symbol
        current_symbols = [chosen_rules[0]]

        # Sequentially apply rules for each layer
        k = 0
        k_2 = 1
        if sample <Pmax:
            rule_types = [0, 1, 0]
        else:
            rule_types = [0, 0, 1]
        
        for layer in range(0, L):  # 1 to 3 (3 layers)
            new_symbols = []
            for symbol in current_symbols:
                rule_type = rule_types[k]
                k = k + 1
                # print(rule_type)
                rule_tensor = rules[layer][rule_type]
                # chosen_rule=torch.randint(low=0,high=rule_tensor.shape[1],size=(1,)).item()
                chosen_rule = chosen_rules[k_2]
                k_2 = k_2 + 1
                new_symbols.extend(rule_tensor[symbol, chosen_rule].tolist())
            # print(new_symbols)
            # new_symbols=new_symbols[0]
            # print(new_symbols)
            if new_symbols != []:
                current_symbols = new_symbols
            features = torch.tensor(new_symbols)
        all_features.append(features)
    concatenated_features = torch.cat(all_features).reshape(len(labels), -1)
    labels = torch.tensor(labels)
    return concatenated_features, labels



def triplets_to_num(xi, n):
    """
    Convert one long input with n-features encoding to n^3 triplet encoding.
    """
    ln = len(xi)
    xin = torch.zeros(ln // 3)
    for ii, xii in enumerate(xi):
        xin[ii // 3] += xii * n ** (2 - ii % 3)
    return xin


def tripling_features(x, n):
    """
    Batch of inputs from n to n^3 encoding using triplets.
    """
    xn = torch.zeros(x.shape[0], x.shape[-1] // 3)
    for i, xi in enumerate(x.squeeze()):
        xn[i] = triplets_to_num(xi, n)
    return xn






def encode_base_v(xi, v):
    """
    Encode a list of digits in base-v to a single integer.
    Most significant digit first.
    """
    encoded = 0
    for i, digit in enumerate(reversed(xi)):
        encoded += digit * (v ** i)
    return encoded

def triplet_pairing_features_T1(x, v):
    """
    Convert inputs of shape [N, 5] into [N, 2] where:
        - output[:, 0] encodes x[:, :3] as base-v triplet -> int in [0, v^3-1]
        - output[:, 1] encodes x[:, 3:5] as base-v pair   -> int in [0, v^2-1]
    """
    N = x.shape[0]
    x_out = torch.zeros(N, 2, dtype=torch.long)

    for i in range(N):
        triplet = x[i, :3]
        pair = x[i, 3:5]
        x_out[i, 0] = encode_base_v(triplet, v)
        x_out[i, 1] = encode_base_v(pair, v)

    return x_out


def triplet_pairing_features_T2(x, v):
    """
    Convert inputs of shape [N, 5] into [N, 2] where:
        - output[:, 0] encodes x[:, :3] as base-v triplet -> int in [0, v^3-1]
        - output[:, 1] encodes x[:, 3:5] as base-v pair   -> int in [0, v^2-1]
    """
    N = x.shape[0]
    x_out = torch.zeros(N, 2, dtype=torch.long)

    for i in range(N):
        triplet = x[i, 2:5]
        pair = x[i, :2]
        x_out[i, 0] = encode_base_v(pair, v)
        x_out[i, 1] = encode_base_v(triplet, v)

    return x_out



def pairs_to_num(xi, n):

    """
        Convert one long input with n-features encoding to n^2 pairs encoding.
    """
    ln = len(xi)
    xin = torch.zeros(ln // 2)
    for ii, xii in enumerate(xi):
        xin[ii // 2] += xii * n ** (1 - ii % 2)
    return xin

def pairing_features(x, n):
    """
        Batch of inputs from n to n^2 encoding.
    """
    xn = torch.zeros(x.shape[0], x.shape[-1] // 2)
    for i, xi in enumerate(x.squeeze()):
        xn[i] = pairs_to_num(xi, n)
    return xn

def two_layers(w1, seed, x, y):
    h, v2 = w1.size()
    assert v2 == x.shape[-1], "Input dim. not matching!"
    v = int(v2 ** .5)

    g = torch.Generator()
    g.manual_seed(seed)
    w2 = torch.randn(v, h, generator=g)

    o = (w2 @ (w1 @ x.t()).div(v2 ** .5).relu() / h).t()

    loss = torch.nn.functional.cross_entropy(o, y, reduction="mean")

    return loss, o




def train_soft_model_T1(features, labels, mask,fac,h,seed_net,temperature,v):
    x = features[mask]
    y = labels[mask]
    x_pair = triplet_pairing_features_T1(x, v)
    x0 = F.one_hot(x_pair[:, 0], num_classes=v ** 3).float()
    x1 = F.one_hot(x_pair[:, 1], num_classes=v ** 2).float()
    
    w0 = torch.ones(h, v ** 3, requires_grad=True)
    loss0, _ = two_layers(w0, seed_net, x0, y)
    loss0.backward()
    grad0 = w0.grad.clone()

    w1 = torch.ones(h, v ** 2, requires_grad=True)
    loss1, _ = two_layers(w1, seed_net, x1, y)
    loss1.backward()
    grad1 = w1.grad.clone()
    
    with torch.no_grad():
        ws0 = nn.Parameter(w0 - fac*h*grad0)
        ws1 = nn.Parameter(w1 - fac*h*grad1)

    kms0 = KMeans(n_clusters=v, n_init=10).fit(ws0.t().detach().numpy())
    kms1 = KMeans(n_clusters=v, n_init=10).fit(ws1.t().detach().numpy())
    
    # Full dataset projection
    x_pair_full = triplet_pairing_features_T1(features, v)
    x0_full = F.one_hot(x_pair_full[:, 0], num_classes=v ** 3).float()
    x1_full = F.one_hot(x_pair_full[:, 1], num_classes=v ** 2).float()

    # Get centroids as torch tensors
    # Project raw one-hot inputs to hidden space using trained weights
    z0 = x0_full @ ws0.T    # [N, h]
    z1 = x1_full @ ws1.T    # [N, h]

    # Convert sklearn centroids to tensors in hidden space
    centroids0 = torch.tensor(kms0.cluster_centers_, dtype=torch.float32)  # [v, h]
    centroids1 = torch.tensor(kms1.cluster_centers_, dtype=torch.float32)  # [v, h]

    def soft_assign(x, centroids, temp):
        dists = torch.cdist(x, centroids)  # [N, v]
        return F.softmax(-dists / temp, dim=1)


    # Compute soft cluster assignments in hidden space
    probs_0 = soft_assign(z0, centroids0, temperature)  # [N, v]
    probs_1 = soft_assign(z1, centroids1, temperature)  # [N, v]


# Compute joint probabilities over v^2 pairs
    # For each data point: outer product of probs_0[i] and probs_1[i]
    probs_joint = torch.einsum('ni,nj->nij', probs_0, probs_1).reshape(-1, v ** 2)  # [N, v^2]
    x=probs_joint.unsqueeze(-1)
    w_top = nn.Parameter(torch.ones(h, v ** 2))
    loss, o = two_layers(w_top, seed_net + 1, x[..., 0], labels)
    loss.backward()
    with torch.no_grad():
        w_top -= fac * h * w_top.grad
        loss, o = two_layers(w_top, seed_net + 1, x[..., 0], labels)
    kms2 = KMeans(n_clusters=v, n_init=10).fit(w_top.t().detach().numpy())

    # x_pair_full=x.squeeze(-1)
    # print(x_pair_full.shape)
    # z=x_pair_full @w_top.T
    # centroids2 = torch.tensor(kms2.cluster_centers_, dtype=torch.float32)  # [v, h]
    # probs_labels = soft_assign(z, centroids2, temperature)  # [N, v]
    # # Compute negative log-probability of the correct label for each sample
    # # probs_labels: [N, v], labels: [N]
    # probs_correct = probs_labels[torch.arange(len(labels)), labels]  # [N]
    per_datum_loss = F.cross_entropy(o, labels, reduction="none")
    # Get predicted labels (argmax over class dimension)
    predicted = o.argmax(dim=1)

    # Compare to true labels and convert to float (1.0 for correct, 0.0 for incorrect)
    per_datum_accuracy = (predicted == labels).float()


    return per_datum_loss,per_datum_accuracy,w_top,ws0,ws1, kms0, kms1, kms2


def train_soft_model_T1_pruned(features, labels, mask,fac,h,seed_net,temperature,v):
    x = features[mask]
    y = labels[mask]
    x_pair = triplet_pairing_features_T1(x, v)
    x0 = F.one_hot(x_pair[:, 0], num_classes=v ** 3).float()
    x1 = F.one_hot(x_pair[:, 1], num_classes=v ** 2).float()
    
    w0 = torch.ones(h, v ** 3, requires_grad=True)
    loss0, _ = two_layers(w0, seed_net, x0, y)
    loss0.backward()
    grad0 = w0.grad.clone()

    w1 = torch.ones(h, v ** 2, requires_grad=True)
    loss1, _ = two_layers(w1, seed_net, x1, y)
    loss1.backward()
    grad1 = w1.grad.clone()
    
    with torch.no_grad():
        ws0 = nn.Parameter(w0 - fac*h*grad0)
        ws1 = nn.Parameter(w1 - fac*h*grad1)

    kms0 = KMeans(n_clusters=v, n_init=10).fit(ws0.t().detach().numpy())
    kms1 = KMeans(n_clusters=v, n_init=10).fit(ws1.t().detach().numpy())

    #Assigned dataset projection

    x0 = F.one_hot(x_pair[:, 0], num_classes=v ** 3).float()
    x1 = F.one_hot(x_pair[:, 1], num_classes=v ** 2).float()

    # Get centroids as torch tensors
    # Project raw one-hot inputs to hidden space using trained weights
    z0 = x0 @ ws0.T    # [N, h]
    z1 = x1 @ ws1.T    # [N, h]

    # Convert sklearn centroids to tensors in hidden space
    centroids0 = torch.tensor(kms0.cluster_centers_, dtype=torch.float32)  # [v, h]
    centroids1 = torch.tensor(kms1.cluster_centers_, dtype=torch.float32)  # [v, h]

    def soft_assign(x, centroids, temp):
        dists = torch.cdist(x, centroids)  # [N, v]
        return F.softmax(-dists / temp, dim=1)


    # Compute soft cluster assignments in hidden space
    probs_0 = soft_assign(z0, centroids0, temperature)  # [N, v]
    probs_1 = soft_assign(z1, centroids1, temperature)  # [N, v]


# Compute joint probabilities over v^2 pairs
    # For each data point: outer product of probs_0[i] and probs_1[i]
    probs_joint= torch.einsum('ni,nj->nij', probs_0, probs_1).reshape(-1, v ** 2)  # [N, v^2]

    x= probs_joint.unsqueeze(-1)

    # Full dataset projection
    x_pair_full = triplet_pairing_features_T1(features, v)
    x0_full = F.one_hot(x_pair_full[:, 0], num_classes=v ** 3).float()
    x1_full = F.one_hot(x_pair_full[:, 1], num_classes=v ** 2).float()

    # Get centroids as torch tensors
    # Project raw one-hot inputs to hidden space using trained weights
    z0 = x0_full @ ws0.T    # [N, h]
    z1 = x1_full @ ws1.T    # [N, h]

    # Convert sklearn centroids to tensors in hidden space
    centroids0 = torch.tensor(kms0.cluster_centers_, dtype=torch.float32)  # [v, h]
    centroids1 = torch.tensor(kms1.cluster_centers_, dtype=torch.float32)  # [v, h]

    def soft_assign(x, centroids, temp):
        dists = torch.cdist(x, centroids)  # [N, v]
        return F.softmax(-dists / temp, dim=1)


    # Compute soft cluster assignments in hidden space
    probs_0 = soft_assign(z0, centroids0, temperature)  # [N, v]
    probs_1 = soft_assign(z1, centroids1, temperature)  # [N, v]


# Compute joint probabilities over v^2 pairs
    # For each data point: outer product of probs_0[i] and probs_1[i]
    probs_joint_full = torch.einsum('ni,nj->nij', probs_0, probs_1).reshape(-1, v ** 2)  # [N, v^2]
    x_full = probs_joint_full.unsqueeze(-1)


    w_top = nn.Parameter(torch.ones(h, v ** 2))
    loss2, _ = two_layers(w_top, seed_net + 1, x[..., 0], y)
    loss2.backward()
    grad2 = w_top.grad.clone()
    with torch.no_grad():
        w_top = nn.Parameter(w_top - fac*h*grad2)

    with torch.no_grad():
        loss, o = two_layers(w_top, seed_net + 1, x_full[..., 0], labels)
    #kms2 = KMeans(n_clusters=v, n_init=10).fit(w_top.t().detach().numpy())

    # x_pair_full=x.squeeze(-1)
    # print(x_pair_full.shape)
    # z=x_pair_full @w_top.T
    # centroids2 = torch.tensor(kms2.cluster_centers_, dtype=torch.float32)  # [v, h]
    # probs_labels = soft_assign(z, centroids2, temperature)  # [N, v]
    # # Compute negative log-probability of the correct label for each sample
    # # probs_labels: [N, v], labels: [N]
    # probs_correct = probs_labels[torch.arange(len(labels)), labels]  # [N]
    per_datum_loss = F.cross_entropy(o, labels, reduction="none")
    # Get predicted labels (argmax over class dimension)
    predicted = o.argmax(dim=1)

    # Compare to true labels and convert to float (1.0 for correct, 0.0 for incorrect)
    per_datum_accuracy = (predicted == labels).float()


    return per_datum_loss,per_datum_accuracy,w_top,ws0,ws1, kms0, kms1

def train_soft_model_T2(features, labels, mask, fac, h, seed_net, temperature, v):
    x = features[mask]
    y = labels[mask]

    # Triplet features for training subset
    x_pair = triplet_pairing_features_T2(x, v)

    # Slot 0
    x0 = F.one_hot(x_pair[:, 0], num_classes=v ** 2).float()
    w0 = torch.ones(h, v ** 2, requires_grad=True)
    loss0, _ = two_layers(w0, seed_net, x0, y)
    loss0.backward()
    grad0 = w0.grad.clone()

    # Slot 1
    x1 = F.one_hot(x_pair[:, 1], num_classes=v ** 3).float()
    w1 = torch.ones(h, v ** 3, requires_grad=True)
    loss1, _ = two_layers(w1, seed_net, x1, y)
    loss1.backward()
    grad1 = w1.grad.clone()

    # Gradient update
    with torch.no_grad():
        ws0 = nn.Parameter(w0 - fac * h * grad0)
        ws1 = nn.Parameter(w1 - fac * h * grad1)

    # KMeans fit
    kms0 = KMeans(n_clusters=v, n_init=10).fit(ws0.t().detach().numpy())
    kms1 = KMeans(n_clusters=v, n_init=10).fit(ws1.t().detach().numpy())

    # Full dataset projection
    x_pair_full = triplet_pairing_features_T2(features, v)
    x0_full = F.one_hot(x_pair_full[:, 0], num_classes=v ** 2).float()
    x1_full = F.one_hot(x_pair_full[:, 1], num_classes=v ** 3).float()

    # Get centroids as torch tensors
    # Project raw one-hot inputs to hidden space using trained weights
    z0 = x0_full @ ws0.T    # [N, h]
    z1 = x1_full @ ws1.T    # [N, h]

    # Convert sklearn centroids to tensors in hidden space
    centroids0 = torch.tensor(kms0.cluster_centers_, dtype=torch.float32)  # [v, h]
    centroids1 = torch.tensor(kms1.cluster_centers_, dtype=torch.float32)  # [v, h]

    def soft_assign(x, centroids, temp):
        dists = torch.cdist(x, centroids)  # [N, v]
        return F.softmax(-dists / temp, dim=1)


    # Compute soft cluster assignments in hidden space
    probs_0 = soft_assign(z0, centroids0, temperature)  # [N, v]
    probs_1 = soft_assign(z1, centroids1, temperature)  # [N, v]


    # Compute joint probabilities over v^2 pairs
    # For each data point: outer product of probs_0[i] and probs_1[i]
    probs_joint = torch.einsum('ni,nj->nij', probs_0, probs_1).reshape(-1, v ** 2)  # [N, v^2]
    x=probs_joint.unsqueeze(-1)
    w_top = nn.Parameter(torch.ones(h, v ** 2))
    loss, o = two_layers(w_top, seed_net + 1, x[..., 0], labels)
    loss.backward()
    with torch.no_grad():
        w_top -= fac * h * w_top.grad
        loss, o = two_layers(w_top, seed_net + 1, x[..., 0], labels)
    kms2 = KMeans(n_clusters=v, n_init=10).fit(w_top.t().detach().numpy())

    # x_pair_full=x.squeeze(-1)
    # print(x_pair_full.shape)
    # z=x_pair_full @w_top.T
    # centroids2 = torch.tensor(kms2.cluster_centers_, dtype=torch.float32)  # [v, h]
    # probs_labels = soft_assign(z, centroids2, temperature)  # [N, v]
    # # Compute negative log-probability of the correct label for each sample
    # # probs_labels: [N, v], labels: [N]
    # probs_correct = probs_labels[torch.arange(len(labels)), labels]  # [N]
    # per_datum_loss = -torch.log(probs_correct + 1e-9)  # avoid log(0)
    per_datum_loss = F.cross_entropy(o, labels, reduction="none")
    # Get predicted labels (argmax over class dimension)
    predicted = o.argmax(dim=1)

    # Compare to true labels and convert to float (1.0 for correct, 0.0 for incorrect)
    per_datum_accuracy = (predicted == labels).float()


    return per_datum_loss,per_datum_accuracy,w_top,ws0,ws1, kms0, kms1, kms2


def train_soft_model_T2_pruned(features, labels, mask, fac, h, seed_net, temperature, v):
    x = features[mask]
    y = labels[mask]

    # Triplet features for training subset
    x_pair = triplet_pairing_features_T2(x, v)

    # Slot 0
    x0 = F.one_hot(x_pair[:, 0], num_classes=v ** 2).float()
    w0 = torch.ones(h, v ** 2, requires_grad=True)
    loss0, _ = two_layers(w0, seed_net, x0, y)
    loss0.backward()
    grad0 = w0.grad.clone()

    # Slot 1
    x1 = F.one_hot(x_pair[:, 1], num_classes=v ** 3).float()
    w1 = torch.ones(h, v ** 3, requires_grad=True)
    loss1, _ = two_layers(w1, seed_net, x1, y)
    loss1.backward()
    grad1 = w1.grad.clone()

    # Gradient update
    with torch.no_grad():
        ws0 = nn.Parameter(w0 - fac * h * grad0)
        ws1 = nn.Parameter(w1 - fac * h * grad1)

    # KMeans fit
    kms0 = KMeans(n_clusters=v, n_init=10).fit(ws0.t().detach().numpy())
    kms1 = KMeans(n_clusters=v, n_init=10).fit(ws1.t().detach().numpy())

    #Assigned dataset projection

    x0 = F.one_hot(x_pair[:, 0], num_classes=v ** 2).float()
    x1 = F.one_hot(x_pair[:, 1], num_classes=v ** 3).float()

    # Get centroids as torch tensors
    # Project raw one-hot inputs to hidden space using trained weights
    z0 = x0 @ ws0.T    # [N, h]
    z1 = x1 @ ws1.T    # [N, h]

    # Convert sklearn centroids to tensors in hidden space
    centroids0 = torch.tensor(kms0.cluster_centers_, dtype=torch.float32)  # [v, h]
    centroids1 = torch.tensor(kms1.cluster_centers_, dtype=torch.float32)  # [v, h]

    def soft_assign(x, centroids, temp):
        dists = torch.cdist(x, centroids)  # [N, v]
        return F.softmax(-dists / temp, dim=1)


    # Compute soft cluster assignments in hidden space
    probs_0 = soft_assign(z0, centroids0, temperature)  # [N, v]
    probs_1 = soft_assign(z1, centroids1, temperature)  # [N, v]


# Compute joint probabilities over v^2 pairs
    # For each data point: outer product of probs_0[i] and probs_1[i]
    probs_joint= torch.einsum('ni,nj->nij', probs_0, probs_1).reshape(-1, v ** 2)  # [N, v^2]

    x= probs_joint.unsqueeze(-1)

    # Full dataset projection
    x_pair_full = triplet_pairing_features_T2(features, v)
    x0_full = F.one_hot(x_pair_full[:, 0], num_classes=v ** 2).float()
    x1_full = F.one_hot(x_pair_full[:, 1], num_classes=v ** 3).float()

    # Get centroids as torch tensors
    # Project raw one-hot inputs to hidden space using trained weights
    z0 = x0_full @ ws0.T    # [N, h]
    z1 = x1_full @ ws1.T    # [N, h]

    # Convert sklearn centroids to tensors in hidden space
    centroids0 = torch.tensor(kms0.cluster_centers_, dtype=torch.float32)  # [v, h]
    centroids1 = torch.tensor(kms1.cluster_centers_, dtype=torch.float32)  # [v, h]

    def soft_assign(x, centroids, temp):
        dists = torch.cdist(x, centroids)  # [N, v]
        return F.softmax(-dists / temp, dim=1)


    # Compute soft cluster assignments in hidden space
    probs_0 = soft_assign(z0, centroids0, temperature)  # [N, v]
    probs_1 = soft_assign(z1, centroids1, temperature)  # [N, v]


# Compute joint probabilities over v^2 pairs
    # For each data point: outer product of probs_0[i] and probs_1[i]
    probs_joint_full = torch.einsum('ni,nj->nij', probs_0, probs_1).reshape(-1, v ** 2)  # [N, v^2]
    x_full = probs_joint_full.unsqueeze(-1)


    w_top = nn.Parameter(torch.ones(h, v ** 2))
    loss2, _ = two_layers(w_top, seed_net + 1, x[..., 0], y)
    loss2.backward()
    grad2 = w_top.grad.clone()
    with torch.no_grad():
        w_top = nn.Parameter(w_top - fac*h*grad2)

    with torch.no_grad():
        loss, o = two_layers(w_top, seed_net + 1, x_full[..., 0], labels)
    #kms2 = KMeans(n_clusters=v, n_init=10).fit(w_top.t().detach().numpy())

    # x_pair_full=x.squeeze(-1)
    # print(x_pair_full.shape)
    # z=x_pair_full @w_top.T
    # centroids2 = torch.tensor(kms2.cluster_centers_, dtype=torch.float32)  # [v, h]
    # probs_labels = soft_assign(z, centroids2, temperature)  # [N, v]
    # # Compute negative log-probability of the correct label for each sample
    # # probs_labels: [N, v], labels: [N]
    # probs_correct = probs_labels[torch.arange(len(labels)), labels]  # [N]
    per_datum_loss = F.cross_entropy(o, labels, reduction="none")
    # Get predicted labels (argmax over class dimension)
    predicted = o.argmax(dim=1)

    # Compare to true labels and convert to float (1.0 for correct, 0.0 for incorrect)
    per_datum_accuracy = (predicted == labels).float()


    return per_datum_loss,per_datum_accuracy,w_top,ws0,ws1, kms0, kms1



def evaluate_combined_model_soft(features_test, labels_test, 
                            kms0_T1, kms1_T1,ws0_T1,ws1_T1, w_top_T1,
                            kms0_T2, kms1_T2,ws0_T2,ws1_T2, w_top_T2,
                            seed_net, v, tree_origin,temperature):
    l = 1
    y = labels_test
    
    def get_logits(features_test, kms0, kms1,ws0,ws1, ws_top, triplet_fn,temperature):
        # Full dataset projection
        x_pair_full = triplet_fn(features_test, v)
        if triplet_fn == triplet_pairing_features_T1:
            x0_full = F.one_hot(x_pair_full[:, 0], num_classes=v ** 3).float()
            x1_full = F.one_hot(x_pair_full[:, 1], num_classes=v ** 2).float()
        else:   
             x0_full = F.one_hot(x_pair_full[:, 0], num_classes=v ** 2).float()
             x1_full = F.one_hot(x_pair_full[:, 1], num_classes=v ** 3).float()


        # Get centroids as torch tensors
        # Project raw one-hot inputs to hidden space using trained weights
        z0 = x0_full @ ws0.T    # [N, h]
        z1 = x1_full @ ws1.T    # [N, h]

        # Convert sklearn centroids to tensors in hidden space
        centroids0 = torch.tensor(kms0.cluster_centers_, dtype=torch.float32)  # [v, h]
        centroids1 = torch.tensor(kms1.cluster_centers_, dtype=torch.float32)  # [v, h]

        def soft_assign(x, centroids, temp):
            dists = torch.cdist(x, centroids)  # [N, v]
            return F.softmax(-dists / temp, dim=1)


        # Compute soft cluster assignments in hidden space
        probs_0 = soft_assign(z0, centroids0, temperature)  # [N, v]
        probs_1 = soft_assign(z1, centroids1, temperature)  # [N, v]


        # Compute joint probabilities over v^2 pairs
        # For each data point: outer product of probs_0[i] and probs_1[i]
        probs_joint = torch.einsum('ni,nj->nij', probs_0, probs_1).reshape(-1, v ** 2)  # [N, v^2]
        #x=probs_joint.unsqueeze(-1)
        with torch.no_grad():
            _, logits = two_layers(ws_top, seed_net + l, probs_joint, y)
        return logits

    # Get logits for both models
    logits_T1 = get_logits(features_test, kms0_T1, kms1_T1,ws0_T1,ws1_T1, w_top_T1, triplet_pairing_features_T1,temperature)
    logits_T2 = get_logits(features_test, kms0_T2, kms1_T2,ws0_T2,ws1_T2, w_top_T2, triplet_pairing_features_T2,temperature)

    # Softmax + confidence
    probs_T1 = F.softmax(logits_T1, dim=1)
    probs_T2 = F.softmax(logits_T2, dim=1)
    conf_T1, preds_T1 = torch.max(probs_T1, dim=1)
    conf_T2, preds_T2 = torch.max(probs_T2, dim=1)

    # Choose most confident prediction
    use_T1 = conf_T1 > conf_T2
    final_preds = torch.where(use_T1, preds_T1, preds_T2)

    # Compute test error
    test_error = (final_preds != y).float().mean().item()
    # 0 = T1, 1 = T2
    predicted_tree = (~use_T1).long()  # use_T1==True → predict T1 (0), otherwise T2 (1)
    tree_error = (predicted_tree != tree_origin).float().mean().item()
    return test_error,tree_error
