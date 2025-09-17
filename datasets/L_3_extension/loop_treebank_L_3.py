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

import argparse





def encode_base_v(xi, v):
    """
    Encode a list of digits in base-v to a single integer.
    Most significant digit first.
    """
    encoded = 0
    for i, digit in enumerate(reversed(xi)):
        encoded += digit * (v ** i)
    return encoded

def triplet_pairing_features_L_3_T1(x, v):
    """
    Convert inputs of shape [N, 9] into [N, 4] where:
        - output[:, 0] encodes x[:, :3] as base-v triplet -> int in [0, v^3-1]
        - output[:, 1] encodes x[:, 3:5] as base-v pair   -> int in [0, v^2-1]
        - output[:, 2] encodes x[:, 5:7] as base-v pair -> int in [0, v^2-1]
        - output[:, 3] encodes x[:, 7:9] as base-v pair -> int in [0, v^2-1]
    """
    N = x.shape[0]
    x_out = torch.zeros(N, 4, dtype=torch.long)

    for i in range(N):
        triplet = x[i, :3]
        pair_1 = x[i, 3:5]
        pair_2 = x[i, 5:7]
        pair_3 = x[i, 7:9]
        x_out[i, 0] = encode_base_v(triplet, v)
        x_out[i, 1] = encode_base_v(pair_1, v)
        x_out[i, 2] = encode_base_v(pair_2, v)
        x_out[i, 3] = encode_base_v(pair_3, v)

    return x_out


def triplet_pairing_features_L_3_T2(x, v):
    """
    Convert inputs of shape [N, 9] into [N, 4] where:
        - output[:, 0] encodes x[:, :2] as base-v pair -> int in [0, v^2-1]
        - output[:, 1] encodes x[:, 2:5] as base-v triplet   -> int in [0, v^3-1]
        - output[:, 2] encodes x[:, 5:7] as base-v pair   -> int in [0, v^2-1]
        - output[:, 3] encodes x[:, 7:9] as base-v pair -> int in [0, v^2-1]
    """
    N = x.shape[0]
    x_out = torch.zeros(N, 4, dtype=torch.long)

    for i in range(N):
        pair_1 = x[i, :2]
        triplet = x[i, 2:5]
        pair_2 = x[i, 5:7]
        pair_3 = x[i, 7:9]
        x_out[i, 0] = encode_base_v(pair_1, v)
        x_out[i, 1] = encode_base_v(triplet, v)
        x_out[i, 2] = encode_base_v(pair_2, v)
        x_out[i, 3] = encode_base_v(pair_3, v)

    return x_out

def triplet_pairing_features_L_3_T3(x, v):
    """
    Convert inputs of shape [N, 9] into [N, 4] where:
        - output[:, 0] encodes x[:, :2] as base-v pair -> int in [0, v^2-1]
        - output[:, 1] encodes x[:, 2:4] as base-v pair   -> int in [0, v^2-1]
        - output[:, 2] encodes x[:, 4:7] as base-v triplet   -> int in [0, v^3-1]
        - output[:, 3] encodes x[:, 7:9] as base-v pair -> int in [0, v^2-1]
    """
    N = x.shape[0]
    x_out = torch.zeros(N, 4, dtype=torch.long)

    for i in range(N):
        pair_1 = x[i, :2]
        pair_2 = x[i, 2:4]
        triplet = x[i, 4:7]
        pair_3 = x[i, 7:9]
        x_out[i, 0] = encode_base_v(pair_1, v)
        x_out[i, 1] = encode_base_v(pair_2, v)
        x_out[i, 2] = encode_base_v(triplet, v)
        x_out[i, 3] = encode_base_v(pair_3, v)

    return x_out


def triplet_pairing_features_L_3_T4(x, v):
    """
    Convert inputs of shape [N, 9] into [N, 4] where:
        - output[:, 0] encodes x[:, :2] as base-v pair -> int in [0, v^2-1]
        - output[:, 1] encodes x[:, 2:4] as base-v pair   -> int in [0, v^2-1]
        - output[:, 2] encodes x[:, 4:6] as base-v pair   -> int in [0, v^2-1]
        - output[:, 3] encodes x[:, 6:9] as base-v triplet -> int in [0, v^3-1]
    """
    N = x.shape[0]
    x_out = torch.zeros(N, 4, dtype=torch.long)

    for i in range(N):
        pair_1 = x[i, :2]
        pair_2 = x[i, 2:4]
        pair_3 = x[i, 4:6]
        triplet = x[i, 6:9]
        x_out[i, 0] = encode_base_v(pair_1, v)
        x_out[i, 1] = encode_base_v(pair_2, v)
        x_out[i, 2] = encode_base_v(pair_3, v)
        x_out[i, 3] = encode_base_v(triplet, v)

    return x_out

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



def sample_data_from_labels_varying_tree_L_3(labels, rules):
    L = len(rules)
    all_features = []
    tree_types = []
    chosen_rule_types = []  # <-- Add this list to store rule type for each datum
    for label in labels:
        current_symbols = [label]
        k = 0
        rule_types = random.choice([[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        # Find which rule type was chosen (the index of 1 in rule_types)
        chosen_rule_type = rule_types.index(1)-3
        chosen_rule_types.append(chosen_rule_type)
        for layer in range(0, L):
            new_symbols = []
            for symbol in current_symbols:
                rule_type = rule_types[k]
                k = k + 1
                rule_tensor = rules[layer][rule_type]
                chosen_rule = torch.randint(
                    low=0, high=rule_tensor.shape[1], size=(1,)
                ).item()
                new_symbols.extend(rule_tensor[symbol, chosen_rule].tolist())
            if new_symbols != []:
                current_symbols = new_symbols
            features = torch.tensor(new_symbols)
        all_features.append(features)
    concatenated_features = torch.cat(all_features).reshape(len(labels), -1)
    return concatenated_features, labels, chosen_rule_types  # <-- Return the list



def train_and_cluster(x, y, w_init, seed, fac, h, v, temperature):
    """
    One gradient step on w, then KMeans on w^T (hidden dim space),
    then soft-assignments from hidden reps to centroids.
    """
    w = torch.ones_like(w_init, requires_grad=True)
    loss, _ = two_layers(w, seed, x, y)
    loss.backward()
    grad = w.grad.clone()

    with torch.no_grad():
        w = nn.Parameter(w - fac * h * grad)

    # Hidden reps for this (training) batch, then detach for independence
    z = (x @ w.T).detach()

    # Cluster in hidden space of w (same as you did)
    kms = KMeans(n_clusters=v, n_init=2).fit(w.t().detach().numpy())
    centroids = torch.tensor(kms.cluster_centers_, dtype=torch.float32)

    # Soft assignments
    dists = torch.cdist(z, centroids)
    probs = F.softmax(-dists / temperature, dim=1).detach()

    return w, kms, centroids, probs


def soft_assign(z, centroids, temperature):
    dists = torch.cdist(z, centroids)
    return F.softmax(-dists / temperature, dim=1)



def get_pairing_fn(variant: str):
    variant = variant.upper()
    if variant == "T1":
        return triplet_pairing_features_L_3_T1, (3, 2, 2, 2)
    if variant == "T2":
        return triplet_pairing_features_L_3_T2, (2, 3, 2, 2)
    if variant == "T3":
        return triplet_pairing_features_L_3_T3, (2, 2, 3, 2)
    if variant == "T4":
        return triplet_pairing_features_L_3_T4, (2, 2, 2, 3)
    raise ValueError(f"Unknown variant {variant}. Expected 'T1'/'T2'/'T3'/'T4'.")


def one_hot_from_pairing(x_pair, pattern, v):
    """
    Build one-hot tensors for the 4 slots according to pattern (tuple of 2/3 per slot).
    Returns a list [x_slot0, x_slot1, x_slot2, x_slot3].
    """
    xs = []
    for slot in range(4):
        deg = pattern[slot]
        num_classes = v ** deg
        xs.append(F.one_hot(x_pair[:, slot], num_classes=num_classes).float())
    return xs  # list of 4 tensors [N, v^deg]


def train_soft_model_L3(features, labels, mask, fac, h, seed_net, temperature, v, variant="T1"):
    """
    Train on masked subset according to topology variant (T1..T4),
    then return losses/acc on *full* dataset along with weights and KMeans objects.
    """
    pairing_fn, pattern = get_pairing_fn(variant)

    # ==== Training subset ====
    x = features[mask]
    y = labels[mask]
    x_pair = pairing_fn(x, v)  # [N_masked, 4] indices
    x_slots = one_hot_from_pairing(x_pair, pattern, v)  # 4 tensors

    # ==== Bottom stage (4 sub-layers) ====
    # For each slot, decide input dim by pattern (2->v^2, 3->v^3)
    w_list = []
    kms_list = []
    cent_list = []
    probs_list = []
    for si, x_s in enumerate(x_slots):
        deg = pattern[si]
        in_dim = v ** deg
        w, kms, cent, probs = train_and_cluster(x_s, y, torch.ones(h, in_dim), seed_net, fac, h, v, temperature)
        w_list.append(w)
        kms_list.append(kms)
        cent_list.append(cent)
        probs_list.append(probs)

    # ==== Intermediate stage (2 joints): (slot0 with slot1), (slot2 with slot3) ====
    probs_joint_1 = torch.einsum("ni,nj->nij", probs_list[0], probs_list[1]).reshape(-1, v ** 2)
    probs_joint_2 = torch.einsum("ni,nj->nij", probs_list[2], probs_list[3]).reshape(-1, v ** 2)

    w1_int, kms1_int, cent1_int, probs1_int = train_and_cluster(
        probs_joint_1, y, torch.ones(h, v ** 2), seed_net + 1, fac, h, v, temperature
    )
    w2_int, kms2_int, cent2_int, probs2_int = train_and_cluster(
        probs_joint_2, y, torch.ones(h, v ** 2), seed_net + 1, fac, h, v, temperature
    )

    probs_joint_int = torch.einsum("ni,nj->nij", probs1_int, probs2_int).reshape(-1, v ** 2)

    # ==== Top stage ====
    w_top, _, _, _ = train_and_cluster(
        probs_joint_int, y, torch.ones(h, v ** 2), seed_net + 2, fac, h, v, temperature
    )

    # ==== Final evaluation on full dataset ====
    probs_joint_int_full = forward_pass_all_L3(
        features, v,
        w_list, (w1_int, w2_int), w_top,
        cent_list, (cent1_int, cent2_int),
        pairing_fn, pattern, temperature
    )

    with torch.no_grad():
        _, o = two_layers(w_top, seed_net + 2, probs_joint_int_full, labels)

    per_datum_loss = F.cross_entropy(o, labels, reduction="none")
    per_datum_accuracy = (o.argmax(dim=1) == labels).float()

    # Unpack for return (keep names similar to your T1 code)
    w1_T, w2_T, w3_T, w4_T = w_list
    kms1, kms2, kms3, kms4 = kms_list
    kms1_int, kms2_int = kms1_int, kms2_int

    return (
        per_datum_loss,
        per_datum_accuracy,
        w_top, w1_T, w2_T, w3_T, w4_T,
        w1_int, w2_int,
        kms1, kms2, kms3, kms4, kms1_int, kms2_int
    )


def forward_pass_all_L3(features, v,
                        w_list, w_int_pair, w_top,
                        cent_list, cent_int_pair,
                        pairing_fn, pattern, temperature):
    """
    Compute the top-level input (probs over v^2) for a dataset given learned params.
    """
    x_pair_full = pairing_fn(features, v)
    x_slots_full = one_hot_from_pairing(x_pair_full, pattern, v)

    # bottom hidden reps + probs
    probs_full = []
    for si in range(4):
        z_s = x_slots_full[si] @ w_list[si].T
        probs_full.append(soft_assign(z_s, cent_list[si], temperature))

    # intermediate joints
    probs_joint_1_full = torch.einsum("ni,nj->nij", probs_full[0], probs_full[1]).reshape(-1, v ** 2)
    probs_joint_2_full = torch.einsum("ni,nj->nij", probs_full[2], probs_full[3]).reshape(-1, v ** 2)

    # intermediate hidden reps + probs
    w1_int, w2_int = w_int_pair
    cent1_int, cent2_int = cent_int_pair

    z1_int_full = probs_joint_1_full @ w1_int.T
    z2_int_full = probs_joint_2_full @ w2_int.T

    probs1_int_full = soft_assign(z1_int_full, cent1_int, temperature)
    probs2_int_full = soft_assign(z2_int_full, cent2_int, temperature)

    # final joint (input to top classifier)
    probs_joint_int_full = torch.einsum("ni,nj->nij", probs1_int_full, probs2_int_full).reshape(-1, v ** 2)
    return probs_joint_int_full



def evaluate_soft_model_L3(features_test, labels_test, v,
                           # weights
                           w1_T, w2_T, w3_T, w4_T, w1_int, w2_int, w_top,
                           # KMeans objects (to recover centroids)
                           kms1, kms2, kms3, kms4, kms1_int, kms2_int,
                           temperature, seed_net, variant="T1"):
    """
    Evaluate trained model on (features_test, labels_test) for a given topology.
    """
    pairing_fn, pattern = get_pairing_fn(variant)

    # Convert KMeans to torch centroids
    cent_list = [
        torch.tensor(kms1.cluster_centers_, dtype=torch.float32),
        torch.tensor(kms2.cluster_centers_, dtype=torch.float32),
        torch.tensor(kms3.cluster_centers_, dtype=torch.float32),
        torch.tensor(kms4.cluster_centers_, dtype=torch.float32),
    ]
    cent_int_pair = (
        torch.tensor(kms1_int.cluster_centers_, dtype=torch.float32),
        torch.tensor(kms2_int.cluster_centers_, dtype=torch.float32),
    )

    w_list = [w1_T, w2_T, w3_T, w4_T]

    # Forward over the test set to build top input
    probs_joint_int_test = forward_pass_all_L3(
        features_test, v,
        w_list, (w1_int, w2_int), w_top,
        cent_list, cent_int_pair,
        pairing_fn, pattern, temperature
    )

    # Final classifier
    with torch.no_grad():
        _, o_test = two_layers(w_top, seed_net + 2, probs_joint_int_test, labels_test)

    per_datum_loss = F.cross_entropy(o_test, labels_test, reduction="none")
    per_datum_accuracy = (o_test.argmax(dim=1) == labels_test).float()
    return per_datum_loss, per_datum_accuracy


# ---- compute logits for each variant on the test set ----
def logits_for_variant(features_test, v, bundle, temperature, seed_net):
    pairing_fn, pattern = get_pairing_fn(bundle["variant"])
    probs_joint_int = forward_pass_all_L3(
        features_test, v,
        bundle["w_list"], bundle["w_int_pair"], bundle["w_top"],
        bundle["cent_list"], bundle["cent_int_pair"],
        pairing_fn, pattern, temperature
    )
    # labels arg is dummy here since we only want logits
    with torch.no_grad():
        _, logits = two_layers(bundle["w_top"], seed_net + 2, probs_joint_int,
                               torch.zeros(len(features_test), dtype=torch.long))
    return logits  # [N, v]



f=1/8
v=16
m2=int(f*v)
m3=int(f*v**2)
s_2=2
s_3=3
n=v
L=3
Pmax=4*f**7*v**9


temperature=1e-5
h=512   
fac=10000

#PP=np.logspace(np.log10(100), np.log10(Pmax-4000), num=12, dtype=int)
PP=[int(Pmax)-4000]

num_realizations=10

results=defaultdict(list)



for P in PP:
    print(f"\n=== Running for P={P} ===")
    vanilla_errors=[]
    vanilla_tree_errors=[]


    for realization in range(num_realizations):
        print(f"\n--- Realization {realization+1} ---")

        # ==== Sample data and rules ====
        max_data = 4*f**7*v**9

        train_size=P
        test_size=2000
        
        labels = torch.randint(0, v, (train_size,))
        labels_test = torch.randint(0, v, (test_size,))

        seed_rules= random.randint(0,10000)
        rules = sample_mixed_rules(v, n, m2, m3, s_2, s_3, L, seed_rules)

        features, labels, topology_ids = sample_data_from_labels_varying_tree_L_3(labels, rules)
        features_test, labels_test, topology_ids_test = sample_data_from_labels_varying_tree_L_3(labels_test, rules)

        N=len(features)

        seed_net= random.randint(0,10000)
        

        topology_ids_tensor = torch.tensor(topology_ids)

        mask_T1 = topology_ids_tensor == 0
        mask_T2 = topology_ids_tensor == 1
        mask_T3 = topology_ids_tensor == 2
        mask_T4 = topology_ids_tensor == 3


        # T1
        (per_datum_loss_T1,
        per_datum_acc_T1,
        w_top_T1, w1_T1, w2_T1, w3_T1, w4_T1,
        w1_int_T1, w2_int_T1,
        kms1_T1, kms2_T1, kms3_T1, kms4_T1, kms1_int_T1, kms2_int_T1) = train_soft_model_L3(
            features, labels, mask_T1, fac, h, seed_net, temperature, v, variant="T1"
        )

        # T2
        (per_datum_loss_T2,
        per_datum_acc_T2,
        w_top_T2, w1_T2, w2_T2, w3_T2, w4_T2,
        w1_int_T2, w2_int_T2,
        kms1_T2, kms2_T2, kms3_T2, kms4_T2, kms1_int_T2, kms2_int_T2) = train_soft_model_L3(
            features, labels, mask_T2, fac, h, seed_net, temperature, v, variant="T2"
        )   

        # T3
        (per_datum_loss_T3,
        per_datum_acc_T3,
        w_top_T3, w1_T3, w2_T3, w3_T3, w4_T3,
        w1_int_T3, w2_int_T3,
        kms1_T3, kms2_T3, kms3_T3, kms4_T3, kms1_int_T3, kms2_int_T3) = train_soft_model_L3(
            features, labels, mask_T3, fac, h, seed_net, temperature, v, variant="T3"
        )

        # T4
        (per_datum_loss_T4,
        per_datum_acc_T4,
        w_top_T4, w1_T4, w2_T4, w3_T4, w4_T4,
        w1_int_T4, w2_int_T4,
        kms1_T4, kms2_T4, kms3_T4, kms4_T4, kms1_int_T4, kms2_int_T4) = train_soft_model_L3(
            features, labels, mask_T4, fac, h, seed_net, temperature, v, variant="T4"
        )



        # ---- pack your four trained models into bundles ----
        bundles = {
            "T1": {
                "w_top": w_top_T1,
                "w_list": [w1_T1, w2_T1, w3_T1, w4_T1],
                "w_int_pair": (w1_int_T1, w2_int_T1),
                "cent_list": [
                    torch.tensor(kms1_T1.cluster_centers_, dtype=torch.float32),
                    torch.tensor(kms2_T1.cluster_centers_, dtype=torch.float32),
                    torch.tensor(kms3_T1.cluster_centers_, dtype=torch.float32),
                    torch.tensor(kms4_T1.cluster_centers_, dtype=torch.float32),
                ],
                "cent_int_pair": (
                    torch.tensor(kms1_int_T1.cluster_centers_, dtype=torch.float32),
                    torch.tensor(kms2_int_T1.cluster_centers_, dtype=torch.float32),
                ),
                "variant": "T1",
            },
            "T2": {
                "w_top": w_top_T2,
                "w_list": [w1_T2, w2_T2, w3_T2, w4_T2],
                "w_int_pair": (w1_int_T2, w2_int_T2),
                "cent_list": [
                    torch.tensor(kms1_T2.cluster_centers_, dtype=torch.float32),
                    torch.tensor(kms2_T2.cluster_centers_, dtype=torch.float32),
                    torch.tensor(kms3_T2.cluster_centers_, dtype=torch.float32),
                    torch.tensor(kms4_T2.cluster_centers_, dtype=torch.float32),
                ],
                "cent_int_pair": (
                    torch.tensor(kms1_int_T2.cluster_centers_, dtype=torch.float32),
                    torch.tensor(kms2_int_T2.cluster_centers_, dtype=torch.float32),
                ),
                "variant": "T2",
            },
            "T3": {
                "w_top": w_top_T3,
                "w_list": [w1_T3, w2_T3, w3_T3, w4_T3],
                "w_int_pair": (w1_int_T3, w2_int_T3),
                "cent_list": [
                    torch.tensor(kms1_T3.cluster_centers_, dtype=torch.float32),
                    torch.tensor(kms2_T3.cluster_centers_, dtype=torch.float32),
                    torch.tensor(kms3_T3.cluster_centers_, dtype=torch.float32),
                    torch.tensor(kms4_T3.cluster_centers_, dtype=torch.float32),
                ],
                "cent_int_pair": (
                    torch.tensor(kms1_int_T3.cluster_centers_, dtype=torch.float32),
                    torch.tensor(kms2_int_T3.cluster_centers_, dtype=torch.float32),
                ),
                "variant": "T3",
            },
            "T4": {
                "w_top": w_top_T4,
                "w_list": [w1_T4, w2_T4, w3_T4, w4_T4],
                "w_int_pair": (w1_int_T4, w2_int_T4),
                "cent_list": [
                    torch.tensor(kms1_T4.cluster_centers_, dtype=torch.float32),
                    torch.tensor(kms2_T4.cluster_centers_, dtype=torch.float32),
                    torch.tensor(kms3_T4.cluster_centers_, dtype=torch.float32),
                    torch.tensor(kms4_T4.cluster_centers_, dtype=torch.float32),
                ],
                "cent_int_pair": (
                    torch.tensor(kms1_int_T4.cluster_centers_, dtype=torch.float32),
                    torch.tensor(kms2_int_T4.cluster_centers_, dtype=torch.float32),
                ),
                "variant": "T4",
            },
        }


        # ---- evaluate all 4 models and select by max logit ----
        VARIANTS = ["T1", "T2", "T3", "T4"]
        N, V = features_test.shape[0], v
        logits_stack = torch.full((N, V, 4), float("-inf"))

        for j, var in enumerate(VARIANTS):
            b = bundles[var]
            logits_stack[:, :, j] = logits_for_variant(features_test, v, b, temperature, seed_net)  # [N, v]

        # per-variant confidence = max logit over classes
        maxlog = logits_stack.max(dim=1).values       # [N, 4]
        best_variant_idx = maxlog.argmax(dim=1)       # [N], index in {0..3}

        # final prediction = argmax over classes in the chosen variant
        pred_each_var = logits_stack.argmax(dim=1)    # [N, 4]
        final_pred = pred_each_var[torch.arange(N), best_variant_idx]  # [N]

        # optional: accuracy if labels_test available
        mean_acc = (final_pred == labels_test).float().mean()
        chosen_names = [VARIANTS[i] for i in best_variant_idx.tolist()]

        error = 1 - mean_acc.item()
        vanilla_errors.append(error)
        print(f"Vanilla test error: {error:.4f}")
        topology_ids_test_tensor = torch.tensor(topology_ids_test)
        vanilla_tree_error = (best_variant_idx != topology_ids_test_tensor).float().mean().item()
        vanilla_tree_errors.append(vanilla_tree_error)
        print(f"Vanilla tree error: {vanilla_tree_error:.4f}")
    
    results["PP"].append(P)
    results["vanilla_error"].append(np.mean(vanilla_errors))
    results["vanilla_error_std"].append(np.std(vanilla_errors))
    results["vanilla_tree_error"].append(np.mean(vanilla_tree_errors))
    results["vanilla_tree_error_std"].append(np.std(vanilla_tree_errors))


with open(f"results_Mallach_Treebank_L_3_f_18_v_16.pkl", "wb") as f:
    pickle.dump(results, f)

print("results saved.")
