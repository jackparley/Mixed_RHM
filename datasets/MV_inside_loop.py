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


from functions_MV_inside import (
    sample_mixed_rules,
    sample_data_from_indices_d_5,
    train_forward_pass,
    train_ternary_patch,
    train_binary_patch,
    train_top_layer,
    evaluate_combined_model_sharing,
    sample_non_overlapping_mixed_rules,
    encode_base_v,
    triplet_pairing_features_T1,
    triplet_pairing_features_T2,
    two_layers,
    soft_assign
   
)




v = 16
f=1/4
m_2=int(f*v)
m_3=int(f*v**2)
s_2=2
s_3=3
L=2
seed_rules=0
n=v
fac=10000
num_features = v
h=1024

temperature_train =1  # Temperature for softmax
temperature_test = 1e-5  # Temperature for softmax

#PP = np.logspace(np.log10(100), np.log10(30000), num=8, dtype=int)
PP=[30000]
results = defaultdict(list)
num_realizations = 10




for P in PP:
    print(f"\n========= Running for P = {P} =========", flush=True)

    single_step_test_errors = []
    single_step_tree_errors = []
    single_step_incorrect_fracs = []

    for realization in range(num_realizations):
        print(f"  -- Realization {realization + 1} --", flush=True)

        max_data = v * m_2**2 * m_3
        train_size = P
        test_size = 2000

        full_range = list(range(2 * max_data))
        samples = random.sample(full_range, train_size)
        remaining_pool = list(set(full_range) - set(samples))
        samples_test = random.sample(remaining_pool, test_size)

        samples = torch.tensor(samples)
        samples_test = torch.tensor(samples_test)
        tree_origin = (samples_test >= max_data).long()
        tree_origin_train = (samples >= max_data).long()

        seed_rules = random.randint(0, 1000000)
        rules = sample_mixed_rules(v, n, m_2, m_3, s_2, s_3, L, seed_rules)

        features, labels = sample_data_from_indices_d_5(samples, rules, n, m_2, m_3)
        features_test, labels_test = sample_data_from_indices_d_5(samples_test, rules, n, m_2, m_3)

        N = len(features)
        assign_T1 = torch.ones(N, dtype=torch.bool)
        assign_T2 = torch.ones(N, dtype=torch.bool)



        seed_net = random.randint(0, 1000000)
        ws0_ter,kms0_ter= train_ternary_patch(features, labels, assign_T1, fac, h, seed_net, v)
        ws0_bin,kms0_bin= train_binary_patch(features, labels, assign_T1, fac, h, seed_net, v)

        # Train top layer
        w_top = train_top_layer(features, labels, assign_T1,assign_T2,
                                fac, h, seed_net, temperature_train, v,
                                ws0_ter, ws0_bin, kms0_ter, kms0_bin)
        # Forward pass
        losses_T1, accs_T1, losses_T2, accs_T2 = train_forward_pass(
            features, labels, seed_net, temperature_train, v,
            ws0_ter, ws0_bin, kms0_ter, kms0_bin, w_top
        )

        new_assign_T1 = losses_T1 < losses_T2
        new_assign_T2 = ~new_assign_T1
        
        assign_T1 = new_assign_T1
        assign_T2 = new_assign_T2

        ws0_ter,kms0_ter= train_ternary_patch(features, labels, assign_T1, fac, h, seed_net, v)
        ws0_bin,kms0_bin= train_binary_patch(features, labels, assign_T2, fac, h, seed_net, v)

         # Train top layer
        w_top = train_top_layer(features, labels, assign_T1,assign_T2,
                                fac, h, seed_net, temperature_test, v,
                                ws0_ter, ws0_bin, kms0_ter, kms0_bin)

        single_step_test_error, single_step_tree_error= evaluate_combined_model_sharing(
            features_test, labels_test, 
            kms0_ter, ws0_ter,
            kms0_bin, ws0_bin, w_top,
            seed_net, v, tree_origin,temperature_test
        )
        
        predicted_tree_train = (assign_T1).long()
        single_step_incorrect_frac = (predicted_tree_train == tree_origin_train).float().mean().item()
        print(f"    Iteration {0}: Test error: {single_step_test_error:.4f}, Tree error: {single_step_tree_error:.4f}, "
              f"Assignment error: {single_step_incorrect_frac:.4f}", flush=True)

        single_step_test_errors.append(single_step_test_error)
        single_step_tree_errors.append(single_step_tree_error)
        single_step_incorrect_fracs.append(single_step_incorrect_frac)


    

    results["P"].append(P)


    results["single_step_error"].append(np.mean(single_step_test_errors))
    results["single_step_error_std"].append(np.std(single_step_test_errors))
    results["single_step_tree_error"].append(np.mean(single_step_tree_errors))
    results["single_step_tree_error_std"].append(np.std(single_step_tree_errors))
    results["single_step_assignment_error"].append(np.mean(single_step_incorrect_fracs))
    results["single_step_assignment_error_std"].append(np.std(single_step_incorrect_fracs))



#outname = f"MV_loop_f_14_soft_T_{args.temperature_train}.pkl"
#with open(outname, "wb") as f:
 #   pickle.dump(results, f)

#print(f"Results saved to {outname}", flush=True)

# Save the results dictionary to a .pkl file
outname = "MV_inside_loop.pkl"
with open(outname, "wb") as f:
    pickle.dump(results, f)

print(f"Results saved to {outname}")

