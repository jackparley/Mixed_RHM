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

from .utils import dec2bin, dec2base, base2dec


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

def index_to_choice(index, n, m2, m3, L, rule_sequence_type):
    # n, m2, m3 = 16, 4, 64  # Number of choices per stage
    if L == 2 and rule_sequence_type == 1:
        index -= 1  # Convert to 0-based index

        x1 = index // (m2 * m3 * m2) + 1
        index = index % (m2 * m3 * m2)

        x2 = index // (m3 * m2) + 1
        index = index % (m3 * m2)

        x3 = index // m2 + 1
        x4 = index % m2 + 1
        choice = [x1, x2, x3, x4]
    elif L == 2 and rule_sequence_type == 2:
        bases = [n, m3, m2, m3, m2]  # Alternating base sizes
        index -= 1  # Convert to 0-based index
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
    elif L == 2 and rule_sequence_type == 3:
        bases = [n, m2, m2, m2]  # Alternating base sizes
        index -= 1  # Convert to 0-based index
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
    elif L == 2 and rule_sequence_type == 4:
        bases = [n, m2, m3, m3]  # Alternating base sizes
        index -= 1  # Convert to 0-based index
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

    elif L == 2 and rule_sequence_type == 5:
        bases = [n, m3, m2, m2, m2]  # Alternating base sizes
        index -= 1  # Convert to 0-based index
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

    elif L == 2 and rule_sequence_type == 6:
        bases = [n, m3, m3, m2, m3]  # Alternating base sizes
        index -= 1  # Convert to 0-based index
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

    elif L == 2 and rule_sequence_type == 7:
        bases = [n, m3, m3, m3, m3]  # Alternating base sizes
        index -= 1  # Convert to 0-based index
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

    elif L == 3:
        bases = [n, m2, m3, m2, m3, m2, m3, m2, m3]  # Alternating base sizes
        index -= 1  # Convert to 0-based index
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
    elif L == 4:
        bases = [
            n,
            m2,
            m3,
            m2,
            m3,
            m2,
            m3,
            m2,
            m3,
            m2,
            m3,
            m2,
            m3,
            m2,
            m3,
            m2,
            m3,
            m2,
            m3,
            m2,
            m3,
            m2,
        ]  # Alternating base sizes
        index -= 1  # Convert to 0-based index
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


def sample_data_from_indices_d_5_check_overlap(
    samples, rules, n, m_2, m_3
):
    L = len(rules)
    Pmax=n*m_2*m_3*m_2
    all_features = []
    labels = []
    overlap_flags = []
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
        
        #for layer in range(0, L):  # 1 to 3 (3 layers)
        layer=0
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

        
        
        layer=1
        rule_tensor = rules[layer][0]
        binary_rule_set = set(map(tuple, rule_tensor.view(-1, 2).tolist()))
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
            if rule_type==1:
                (a, b, c)=rule_tensor[symbol, chosen_rule].tolist()
                if (a, b) in binary_rule_set or (b, c) in binary_rule_set:
                    #print("Binary overlap")
                    overlap_flags.append(1)
                else:
                    overlap_flags.append(0)

        # print(new_symbols)
        # new_symbols=new_symbols[0]
        # print(new_symbols)
        if new_symbols != []:
            current_symbols = new_symbols
        features = torch.tensor(new_symbols)


        all_features.append(features)
    concatenated_features = torch.cat(all_features).reshape(len(labels), -1)
    labels = torch.tensor(labels)
    print("fraction with overlaps:", sum(overlap_flags)/len(overlap_flags))
    return concatenated_features, labels, overlap_flags

def sample_data_from_indices_fixed_tree(
    samples, rules, rule_types, n, m_2, m_3, rule_sequence_type
):
    L = len(rules)
    all_features = []
    labels = []
    #samples = samples + 1
    for sample in samples:
        chosen_rules = index_to_choice(sample, n, m_2, m_3, L, rule_sequence_type)
        labels.append(chosen_rules[0] - 1)
        # print(chosen_rules[0])
        chosen_rules = [x - 1 for x in chosen_rules]
        # for label in labels:
        # Initialize the current symbols with the start symbol
        current_symbols = [chosen_rules[0]]

        # Sequentially apply rules for each layer
        k = 0
        k_2 = 1
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


def sample_padded_rules(v, n, m_2, m_3, s_2, s_3, L, seed):
    random.seed(seed)

    # Define a callable for the defaultdict to return empty tensors
    def tensor_default():
        return torch.empty(0)

    tuples_2 = list(product(*[range(v) for _ in range(s_2)]))
    tuples_3 = list(product(*[range(v) for _ in range(s_3)]))

    # Define the defaultdict structure
    rules = defaultdict(tensor_default)

    # Initialize the grammar with sampled tensors
    binary_rules = torch.tensor(random.sample(tuples_2, n * m_2)).reshape(n, m_2, s_2)
    ternary_rules = torch.tensor(random.sample(tuples_3, n * m_3)).reshape(n, m_3, s_3)

    # Pad binary rules with the fake symbol (integer v)
    padding = torch.full((n, m_2, s_3 - s_2), v)
    padded_binary_rules = torch.cat((binary_rules, padding), dim=2)

    # Stack binary and ternary rules on top of each other
    rules[0] = torch.cat((padded_binary_rules, ternary_rules), dim=1)

    for i in range(1, L):
        binary_rules = torch.tensor(random.sample(tuples_2, v * m_2)).reshape(
            v, m_2, s_2
        )
        ternary_rules = torch.tensor(random.sample(tuples_3, v * m_3)).reshape(
            v, m_3, s_3
        )

        # Pad binary rules with the fake symbol (integer v)
        padding = torch.full((v, m_2, s_3 - s_2), v)
        padded_binary_rules = torch.cat((binary_rules, padding), dim=2)

        # Stack binary and ternary rules on top of each other
        rules[i] = torch.cat((padded_binary_rules, ternary_rules), dim=1)

    # Add the fake symbol with rules [v, v, v] at each layer
    for l in range(L):
        fake_rules = torch.full((1, m_2 + m_3, 3), v)
        rules[l] = torch.cat((rules[l], fake_rules), dim=0)

    return rules

def sample_non_overlapping_padded_rules(v, n, m_2, m_3, s_2, s_3, L, seed):
    random.seed(seed)

    # Define a callable for the defaultdict to return empty tensors
    def tensor_default():
        return torch.empty(0)

    tuples_2 = list(product(*[range(v) for _ in range(s_2)]))
    tuples_3 = list(product(*[range(v) for _ in range(s_3)]))

    # Define the defaultdict structure
    rules = defaultdict(tensor_default)

    binary_rules_tuples = random.sample(tuples_2, n * m_2)
    binary_rule_set = set(binary_rules_tuples)  # for fast lookup
    binary_rules = torch.tensor(binary_rules_tuples).reshape(n, m_2, s_2)

    # Rejection sampling for ternary rules
    valid_ternary_rules = []
    used_ternary_rules = set()  # to avoid replacement

    for t in random.sample(tuples_3, len(tuples_3)):  # shuffle once to ensure uniformity
        if len(valid_ternary_rules) >= n * m_3:
            break
        (a, b, c) = t
        if (a, b) in binary_rule_set or (b, c) in binary_rule_set:
            print("Binary overlap")
            print(t)
            continue
        if t in used_ternary_rules:
            continue
        valid_ternary_rules.append(t)
        used_ternary_rules.add(t)

    # Check if enough ternary rules were found
    if len(valid_ternary_rules) < n * m_3:
        raise ValueError("Not enough valid ternary rules found without binary overlap.")

    # Final tensor
    ternary_rules = torch.tensor(valid_ternary_rules).reshape(n, m_3, s_3)
   
    # Pad binary rules with the fake symbol (integer v)
    padding = torch.full((n, m_2, s_3 - s_2), v)
    padded_binary_rules = torch.cat((binary_rules, padding), dim=2)

    # Stack binary and ternary rules on top of each other
    rules[0] = torch.cat((padded_binary_rules, ternary_rules), dim=1)



    binary_rules_tuples = random.sample(tuples_2, n * m_2)
    binary_rule_set = set(binary_rules_tuples)  # for fast lookup
    binary_rules = torch.tensor(binary_rules_tuples).reshape(n, m_2, s_2)

    # Rejection sampling for ternary rules
    valid_ternary_rules = []
    used_ternary_rules = set()  # to avoid replacement

    for t in random.sample(tuples_3, len(tuples_3)):  # shuffle once to ensure uniformity
        if len(valid_ternary_rules) >= n * m_3:
            break
        (a, b, c) = t
        if (a, b) in binary_rule_set or (b, c) in binary_rule_set:
            continue
        if t in used_ternary_rules:
            continue
        valid_ternary_rules.append(t)
        used_ternary_rules.add(t)

    # Check if enough ternary rules were found
    if len(valid_ternary_rules) < n * m_3:
        raise ValueError("Not enough valid ternary rules found without binary overlap.")

    # Final tensor
    ternary_rules = torch.tensor(valid_ternary_rules).reshape(n, m_3, s_3)
   
    # Pad binary rules with the fake symbol (integer v)
    padding = torch.full((n, m_2, s_3 - s_2), v)
    padded_binary_rules = torch.cat((binary_rules, padding), dim=2)

    # Stack binary and ternary rules on top of each other
    rules[1] = torch.cat((padded_binary_rules, ternary_rules), dim=1)





    # Add the fake symbol with rules [v, v, v] at each layer
    for l in range(L):
        fake_rules = torch.full((1, m_2 + m_3, 3), v)
        rules[l] = torch.cat((rules[l], fake_rules), dim=0)

    return rules

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


def sample_data_from_labels_varying_tree_d_5_4(labels, rules, num_features, d_max):
    L = len(rules)
    all_features = []
    tree_types = []
    for label in labels:
        # Initialize the current symbols with the start symbol
        current_symbols = [label]
        k=0
        rule_types = random.choice([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        #rule_types = random.choice([[0, 0, 0], [0, 1, 0], [0, 0, 1],[0, 1, 1]])
        # Sequentially apply rules for each layer
        for layer in range(0, L):  # 1 to 3 (3 layers)
            new_symbols = []
            
            for symbol in current_symbols:
                #rule_type = torch.randint(low=0, high=2, size=(1,)).item()
                rule_type=rule_types[k]
                k=k+1
                if layer==0:
                    first_rule=rule_type
                # print(rule_type)
                rule_tensor = rules[layer][rule_type]
                chosen_rule = torch.randint(
                    low=0, high=rule_tensor.shape[1], size=(1,)
                ).item()
                new_symbols.extend(rule_tensor[symbol, chosen_rule].tolist())
            # print(new_symbols)
            # new_symbols=new_symbols[0]
            # print(new_symbols)
            if new_symbols != []:
                current_symbols = new_symbols
            features = torch.tensor(new_symbols)
            # Pad this with extra symbol according to d_max
        if len(features) < 6:
            tree_types.append(len(features)-4)
        elif len(features) >6:
            tree_types.append(len(features)-3)
        elif len(features) == 6:
            if first_rule==0:
                tree_types.append(2)
            else:
                tree_types.append(3)
        if len(features) < d_max:
            features = torch.cat(
                [
                    features,
                    num_features
                    * torch.ones(d_max - len(features), dtype=torch.int64),
                ]
            )
        all_features.append(features)
        

        

    concatenated_features = torch.cat(all_features).reshape(len(labels), -1)
    return concatenated_features, labels,tree_types


def sample_data_from_labels_varying_tree_top_ter(labels, rules, num_features, d_max):
    L = len(rules)
    all_features = []
    tree_types = []
    for label in labels:
        while True:  # Keep trying until we get a sequence length not equal to 6
            current_symbols = [label]
            new_symbols = []

            # --- Layer 0 ---
            layer = 0
            for symbol in current_symbols:
                rule_type = 1  # Or any deterministic value or strategy
                rule_tensor = rules[layer][rule_type]
                chosen_rule = torch.randint(low=0, high=rule_tensor.shape[1], size=(1,)).item()
                new_symbols.extend(rule_tensor[symbol, chosen_rule].tolist())

            if new_symbols != []:
                current_symbols = new_symbols
            #features = torch.tensor(new_symbols)

            # --- Layer 1 ---
            layer = 1
            for symbol in current_symbols:
                rule_type = torch.randint(low=0, high=2, size=(1,)).item()
                rule_tensor = rules[layer][rule_type]
                chosen_rule = torch.randint(low=0, high=rule_tensor.shape[1], size=(1,)).item()
                new_symbols.extend(rule_tensor[symbol, chosen_rule].tolist())

            if new_symbols != []:
                current_symbols = new_symbols
            features = torch.tensor(new_symbols)

            d = len(features)

            if d == 6:
                continue  # Reject and sample from scratch
            else:
                break  # Accept this sequence and move to the next label

    # Now `features` has a valid sequence (d != 6), and you can proceed

        tree_types.append(len(features)-3)
        if len(features) < d_max:
            features = torch.cat(
                [
                    features,
                    num_features
                    * torch.ones(d_max - len(features), dtype=torch.int64),
                ]
            )
        all_features.append(features)
        

        

    concatenated_features = torch.cat(all_features).reshape(len(labels), -1)
    return concatenated_features, labels,tree_types



def create_probabilities(m_2, m_3, L):
    probabilities = {}
    for l in range(L):
        prob = torch.cat(
            (torch.full((m_2,), 1 / (2 * m_2)), torch.full((m_3,), 1 / (2 * m_3)))
        )
        probabilities[l] = prob
    return probabilities


def create_probabilities_eta(m_2, m_3, L,eta ):
    probabilities = {}
    p_3=1/(m_2*eta+m_3)
    p_2=eta*p_3
    for l in range(L):
        prob = torch.cat(
            (torch.full((m_2,), p_2), torch.full((m_3,), p_3))
        )
        probabilities[l] = prob
    return probabilities



def sample_data_from_labels_varying_tree_tensorized_d_values(
    labels, rules, probability, num_features, d_max,m_2
):
    """
    Create data of the Random Hierarchy Model starting from class labels and a set of rules. Rules are chosen according to probability.

    Args:
        labels: A tensor of size [batch_size, I], with I from 0 to num_classes-1 containing the class labels of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.
        probability: A dictionary containing the distribution of the rules for each level of the hierarchy.

    Returns:
        A tuple containing the inputs and outputs of the model.
    """
    L = len(rules)  # Number of levels in the hierarchy

    features = labels

    chosen_rule = torch.multinomial(
        probability[0], features.numel(), replacement=True
    ).reshape(
        features.shape
    )  # Choose a rule for each variable in the current level according to probability[l]
    features = rules[0][features, chosen_rule].flatten(
        start_dim=1
    )  # Apply the chosen rule to each variable in the current level

    result_tensor = torch.where((chosen_rule[:] >= 0) & (chosen_rule[:] < m_2), 0, 1)

    for l in range(1,L):
        chosen_rule = torch.multinomial(
            probability[l], features.numel(), replacement=True
        ).reshape(
            features.shape
        )  # Choose a rule for each variable in the current level according to probability[l]
        features = rules[l][features, chosen_rule].flatten(
            start_dim=1
        )  # Apply the chosen rule to each variable in the current level
    mask = features == num_features

    # Count the number of fake symbols in each row
    num_fake_symbols = mask.sum(dim=1)

    # Create a tensor to hold the final result
    result = torch.zeros_like(features)
    num_data = features.shape[0]  # Number of data points
    # Remove the fake symbols and keep the original order of the rest of the elements
    real_features = [features[i, ~mask[i]] for i in range(num_data)]
    # print(real_features)
    # Fill the result tensor with real features and append fake symbols at the end
    tree_types=[]
    for i in range(num_data):
        num_real_symbols = d_max - num_fake_symbols[i]
        result[i, :num_real_symbols] = real_features[i]
        result[i, num_real_symbols:] = num_features
        if num_real_symbols < 6:
            tree_types.append(num_real_symbols.item()-4)
        elif num_real_symbols >6:
            tree_types.append(num_real_symbols.item()-3)
        elif num_real_symbols == 6:
            if result_tensor[i]==0:
                tree_types.append(2)
            else:
                tree_types.append(3)    
    return result, labels,tree_types


def sample_data_from_labels_varying_tree_tensorized(
    labels, rules, probability, num_features, d_max
):
    """
    Create data of the Random Hierarchy Model starting from class labels and a set of rules. Rules are chosen according to probability.

    Args:
        labels: A tensor of size [batch_size, I], with I from 0 to num_classes-1 containing the class labels of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.
        probability: A dictionary containing the distribution of the rules for each level of the hierarchy.

    Returns:
        A tuple containing the inputs and outputs of the model.
    """
    L = len(rules)  # Number of levels in the hierarchy

    features = labels

    for l in range(L):
        chosen_rule = torch.multinomial(
            probability[l], features.numel(), replacement=True
        ).reshape(
            features.shape
        )  # Choose a rule for each variable in the current level according to probability[l]
        features = rules[l][features, chosen_rule].flatten(
            start_dim=1
        )  # Apply the chosen rule to each variable in the current level
    mask = features == num_features

    # Count the number of fake symbols in each row
    num_fake_symbols = mask.sum(dim=1)

    # Create a tensor to hold the final result
    result = torch.zeros_like(features)
    num_data = features.shape[0]  # Number of data points
    # Remove the fake symbols and keep the original order of the rest of the elements
    real_features = [features[i, ~mask[i]] for i in range(num_data)]
    # print(real_features)
    # Fill the result tensor with real features and append fake symbols at the end
    for i in range(num_data):
        num_real_symbols = d_max - num_fake_symbols[i]
        result[i, :num_real_symbols] = real_features[i]
        result[i, num_real_symbols:] = num_features
    return result, labels


def sample_data_from_labels_varying_tree_tensorized_top_ter_d_values(
    labels, rules, probability, num_features, d_max,m_2,m_3
):
    """
    Create data of the Random Hierarchy Model starting from class labels and a set of rules. Rules are chosen according to probability.

    Args:
        labels: A tensor of size [batch_size, I], with I from 0 to num_classes-1 containing the class labels of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.
        probability: A dictionary containing the distribution of the rules for each level of the hierarchy.

    Returns:
        A tuple containing the inputs and outputs of the model.
    """
    L = len(rules)  # Number of levels in the hierarchy

    features = labels

    l=0
    chosen_rule = torch.randint(
    low=m_2, 
    high=m_2 + m_3, 
    size=features.shape, 
    )
    features = rules[l][features, chosen_rule].flatten(
        start_dim=1
    )  # Apply the chosen rule to each variable in the current level
    l=1
    chosen_rule = torch.multinomial(
        probability[l], features.numel(), replacement=True
    ).reshape(
        features.shape
    )  # Choose a rule for each variable in the current level according to probability[l]
    features = rules[l][features, chosen_rule].flatten(
        start_dim=1
    )  # Apply the chosen rule to each variable in the current level


    mask = features == num_features

    # Count the number of fake symbols in each row
    num_fake_symbols = mask.sum(dim=1)

    # Create a tensor to hold the final result
    result = torch.zeros_like(features)
    num_data = features.shape[0]  # Number of data points
    # Remove the fake symbols and keep the original order of the rest of the elements
    real_features = [features[i, ~mask[i]] for i in range(num_data)]
    # print(real_features)
    # Fill the result tensor with real features and append fake symbols at the end
    tree_types=[]
    for i in range(num_data):
        num_real_symbols = d_max - num_fake_symbols[i]
        result[i, :num_real_symbols] = real_features[i]
        result[i, num_real_symbols:] = num_features
        tree_types.append(num_real_symbols.item()-3) 
    return result, labels,tree_types


def sample_data_from_labels_varying_tree_tensorized_padding_central(
    labels, rules, probability, num_features, d_max
):
    """
    Create data of the Random Hierarchy Model starting from class labels and a set of rules. Rules are chosen according to probability.

    Args:
        labels: A tensor of size [batch_size, I], with I from 0 to num_classes-1 containing the class labels of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.
        probability: A dictionary containing the distribution of the rules for each level of the hierarchy.

    Returns:
        A tuple containing the inputs and outputs of the model.
    """
    L = len(rules)  # Number of levels in the hierarchy

    features = labels

    for l in range(L):
        chosen_rule = torch.multinomial(
            probability[l], features.numel(), replacement=True
        ).reshape(
            features.shape
        )  # Choose a rule for each variable in the current level according to probability[l]
        features = rules[l][features, chosen_rule].flatten(
            start_dim=1
        )  # Apply the chosen rule to each variable in the current level
    mask = features == num_features

    # Count the number of fake symbols in each row
    num_fake_symbols = mask.sum(dim=1)

    # Create a tensor to hold the final result
    result = torch.zeros_like(features)
    num_data = features.shape[0]  # Number of data points
    # Remove the fake symbols and keep the original order of the rest of the elements
    real_features = [features[i, ~mask[i]] for i in range(num_data)]
    # print(real_features)
    # Fill the result tensor with real features and append fake symbols at the end
    for i in range(num_data):
        num_real_symbols = d_max - num_fake_symbols[i]
        left_fake = num_fake_symbols[i] // 2 + (
            num_fake_symbols[i] % 2
        )  # Extra fake on left if odd

        result[i, left_fake : left_fake + num_real_symbols] = real_features[i]

    return result, labels


def sample_data_from_labels_fixed_tree(labels, rules, rule_types):
    L = len(rules)
    all_features = []
    for label in labels:
        # Initialize the current symbols with the start symbol
        current_symbols = [label]
        # Sequentially apply rules for each layer
        k = 0
        for layer in range(0, L):  # 1 to 3 (3 layers)
            new_symbols = []
            for symbol in current_symbols:
                rule_type = rule_types[k]
                k = k + 1
                # print(rule_type)
                rule_tensor = rules[layer][rule_type]
                chosen_rule = torch.randint(
                    low=0, high=rule_tensor.shape[1], size=(1,)
                ).item()
                new_symbols.extend(rule_tensor[symbol, chosen_rule].tolist())
            # print(new_symbols)
            # new_symbols=new_symbols[0]
            # print(new_symbols)
            if new_symbols != []:
                current_symbols = new_symbols
            features = torch.tensor(new_symbols)
        all_features.append(features)
    concatenated_features = torch.cat(all_features).reshape(len(labels), -1)
    return concatenated_features, labels


def reconstruct_tree_structure(rule_types, n, m_2, m_3, L):
    tree_structure = []
    current_level = [rule_types[0]]  # Start with the root node
    index = 1

    for level in range(1, L + 1):
        next_level = []
        for node in current_level:
            branching_factor = 3 if node == 1 else 2
            next_level.extend(rule_types[index : index + branching_factor])
            index += branching_factor
        tree_structure.append(current_level)
        current_level = next_level
    last_rules = tree_structure[-1]
    total_sum = sum(3 if x == 1 else 2 for x in last_rules)
    input_size = total_sum
    factor = n

    # Iterate through the nested list and update the factor
    for level in tree_structure:
        for value in level:
            if value == 1:
                factor *= m_3
            elif value == 0:
                factor *= m_2
    num_data = factor
    return tree_structure, input_size, num_data


def sample_rules(v, n, m, s, L, seed=42):
    """
    Sample random rules for a random hierarchy model.

    Args:
        v: The number of values each variable can take (vocabulary size, int).
        n: The number of classes (int).
        m: The number of synonymic lower-level representations (multiplicity, int).
        s: The size of lower-level representations (int).
        L: The number of levels in the hierarchy (int).
        seed: Seed for generating the rules.

    Returns:
        A dictionary containing the rules for each level of the hierarchy.
    """
    random.seed(seed)
    tuples = list(product(*[range(v) for _ in range(s)]))

    rules = {}
    rules[0] = torch.tensor(random.sample(tuples, n * m)).reshape(n, m, -1)
    for i in range(1, L):
        rules[i] = torch.tensor(random.sample(tuples, v * m)).reshape(v, m, -1)

    return rules


def sample_data_from_labels(labels, rules, probability):
    """
    Create data of the Random Hierarchy Model starting from class labels and a set of rules. Rules are chosen according to probability.

    Args:
        lables: A tensor of size [batch_size, I], with I from 0 to num_classes-1 containing the class labels of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.
        probability: A dictionary containing the distribution of the rules for each level of the hierarchy.

    Returns:
        A tuple containing the inputs and outputs of the model.
    """
    L = len(rules)  # Number of levels in the hierarchy

    features = labels

    for l in range(L):
        chosen_rule = torch.multinomial(
            probability[l], features.numel(), replacement=True
        ).reshape(
            features.shape
        )  # Choose a rule for each variable in the current level according to probability[l]
        features = rules[l][features, chosen_rule].flatten(
            start_dim=1
        )  # Apply the chosen rule to each variable in the current level

    return features, labels


def sample_data_from_labels_unif(labels, rules):
    """
    Create data of the Random Hierarchy Model starting from class labels and a set of rules. Rules are chosen Chetuormly at random for each level.

    Args:
        lables: A tensor of size [batch_size, I], with I from 0 to num_classes-1 containing the class labels of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.

    Returns:
        A tuple containing the inputs and outputs of the model.
    """
    L = len(rules)  # Number of levels in the hierarchy

    features = labels

    for l in range(L):
        chosen_rule = torch.randint(
            low=0, high=rules[l].shape[1], size=features.shape
        )  # Choose a random rule for each variable in the current level
        features = rules[l][features, chosen_rule].flatten(
            start_dim=1
        )  # Apply the chosen rule to each variable in the current level
    return features, labels


def sample_data_from_indices(samples, rules, v, n, m, s, L, bonus):
    """
    Create data of the Random Hierarchy Model starting from a set of rules and the sampled indices.

    Args:
        samples: A tensor of size [batch_size, I], with I from 0 to max_data-1, containing the indices of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.
        n: The number of classes (int).
        m: The number of synonymic lower-level representations (multiplicity, int).
        s: The size of lower-level representations (int).
        L: The number of levels in the hierarchy (int).
        bonus: Dictionary for additional output (list), includes 'noise' (randomly replace one symbol at each level), 'synonyms' (randomply resample one production rule at each level), 'tree' (stores the data derivation), 'size' (number of bonus data).

    Returns:
        A tuple containing the inputs and outputs of the model (plus additional output in bonus dict).
    """
    max_data = n * m ** ((s**L - 1) // (s - 1))
    data_per_hl = max_data // n  # div by num_classes to get number of data per class

    high_level = samples.div(
        data_per_hl, rounding_mode="floor"
    )  # div by data_per_hl to get class index (run in range(n))
    low_level = samples % data_per_hl  # compute remainder (run in range(data_per_hl))

    labels = high_level  # labels are the classes (features of highest level)
    features = labels  # init input features as labels (rep. size 1)
    size = 1

    if bonus:  # extra output for additional measures
        if "size" not in bonus.keys():
            bonus["size"] = samples.size(0)
        if "tree" in bonus:
            tree = {}
            bonus["tree"] = tree
        if "noise" in bonus:  # add corrupted versions of the last bonus[-1] data
            noise = {}
            noise[L] = copy.deepcopy(
                features[-bonus["size"] :]
            )  # copy current representation (labels)...
            noise[L][:] = torch.randint(
                n, (bonus["size"],)
            )  # ...and randomly change it
            bonus["noise"] = noise
        if "synonyms" in bonus:  # add synonymic versions of the last bonus[-1] data
            synonyms = {}
            bonus["synonyms"] = synonyms

    for l in range(L):

        choices = m ** (size)
        data_per_hl = (
            data_per_hl // choices
        )  # div by num_choices to get number of data per high-level feature

        high_level = low_level.div(
            data_per_hl, rounding_mode="floor"
        )  # div by data_per_hl to get high-level feature index (1 index in range(m**size))
        high_level = dec2base(
            high_level, m, length=size
        ).squeeze()  # convert to base m (size indices in range(m), squeeze needed if index already in base m)

        if bonus:
            if "tree" in bonus:
                tree[L - l] = copy.deepcopy(features[-bonus["size"] :])

            if "synonyms" in bonus:

                for (
                    ell
                ) in (
                    synonyms.keys()
                ):  # propagate modified data down the tree TODO: randomise whole downstream propagation
                    synonyms[ell] = rules[l][
                        synonyms[ell], high_level[-bonus["size"] :]
                    ]
                    synonyms[ell] = synonyms[ell].flatten(start_dim=1)

                high_level_syn = copy.deepcopy(
                    high_level[-bonus["size"] :]
                )  # copy current representation indices...
                if l == 0:
                    high_level_syn[:] = torch.randint(
                        m, (high_level_syn.size(0),)
                    )  # ... and randomly change it (only one index at the highest level)
                else:
                    high_level_syn[:, -2] = torch.randint(
                        m, (high_level_syn.size(0),)
                    )  # ... and randomly change the next-to-last
                synonyms[L - l] = copy.deepcopy(features[-bonus["size"] :])
                synonyms[L - l] = rules[l][synonyms[L - l], high_level_syn]
                synonyms[L - l] = synonyms[L - l].flatten(start_dim=1)
                # TODO: add custom positions for 'synonyms'

        features = rules[l][
            features, high_level
        ]  # apply l-th rule to expand to get features at the lower level (tensor of size (batch_size, size, s))
        features = features.flatten(
            start_dim=1
        )  # flatten to tensor of size (batch_size, size*s)
        size *= s  # rep. size increases by s at each level
        low_level = (
            low_level % data_per_hl
        )  # compute remainder (run in range(data_per_hl))

        if bonus:
            if "noise" in bonus:

                for (
                    ell
                ) in (
                    noise.keys()
                ):  # propagate modified data down the tree TODO: randomise whole downstream propagation
                    noise[ell] = rules[l][noise[ell], high_level[-bonus["size"] :]]
                    noise[ell] = noise[ell].flatten(start_dim=1)

                noise[L - l - 1] = copy.deepcopy(
                    features[-bonus["size"] :]
                )  # copy current representation ...
                noise[L - l - 1][:, -2] = torch.randint(
                    v, (bonus["size"],)
                )  # ... and randomly change the next-to-last feature
                # TODO: add custom positions for 'noise'

    return features, labels


class MixedRandomHierarchyModel_varying_tree(Dataset):
    """
    Implement the Random Hierarchy Model (RHM) as a PyTorch dataset.
    """

    def __init__(
        self,
        num_features=8,  # vocabulary size
        num_classes=2,  # number of classes
        fraction_rules=1,  # number of synonymic low-level representations (multiplicity)
        s_2=2,
        s_3=3,  # size of the low-level representations
        num_layers=2,
        seed_rules=0,
        seed_sample=1,
        train_size=-1,
        test_size=0,
        d_5_set=0,
        d_5_4_set=0,
        eta_set=0,
        eta=1,
        padding_tail=0,
        padding_central=0,
        return_type=0,
        non_overlapping=0,
        top_ter=0,
        input_format="onehot",
        whitening=0,
        transform=None,
    ):

        v = num_features
        f = fraction_rules/v
        m_2 = int(f * v)
        m_3 = int(f * v**2)
        self.num_features = num_features
        self.m_2 = m_2
        self.m_3 = m_3
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.s_2 = s_2
        self.s_3 = s_3
        self.fraction_rules = fraction_rules
        eta=v/eta
        self.eta=eta
        
        if return_type==1 and d_5_4_set==0 and non_overlapping==0 and top_ter==0:
            #self.rules = sample_mixed_rules(
             #   num_features, num_classes, m_2, m_3, s_2, s_3, num_layers, seed_rules
            #)
            self.rules = sample_padded_rules(
                num_features, num_classes, m_2, m_3, s_2, s_3, num_layers, seed_rules
            )
        elif non_overlapping==1:
            self.rules = sample_non_overlapping_padded_rules(
                num_features, num_classes, m_2, m_3, s_2, s_3, num_layers, seed_rules
            )

        elif d_5_set==1 or d_5_4_set==1:
            self.rules = sample_mixed_rules(
                num_features, num_classes, m_2, m_3, s_2, s_3, num_layers, seed_rules
            )
        elif top_ter==1 and return_type==1:
            self.rules = sample_mixed_rules(
                num_features, num_classes, m_2, m_3, s_2, s_3, num_layers, seed_rules
            )
        else:
            self.rules = sample_padded_rules(
                num_features, num_classes, m_2, m_3, s_2, s_3, num_layers, seed_rules
            )
        

        # tree_structure,input_size,max_data=reconstruct_tree_structure(rule_types,num_classes,m_2,m_3,num_layers)

        torch.manual_seed(seed_sample)
        labels = torch.randint(low=0, high=num_classes, size=(train_size + test_size,))
        if num_layers == 2:
            d_max = 9
        elif num_layers == 3:
            d_max = 27
        elif num_layers == 4:
            d_max = 81
        if padding_central:
            self.features, self.labels = (
                sample_data_from_labels_varying_tree_tensorized_padding_central(
                    labels,
                    self.rules,
                    create_probabilities(m_2, m_3, num_layers),
                    num_features,
                    d_max,
                )
            )
        elif return_type and d_5_4_set==0 and top_ter==0 and eta_set==0:
            #self.features, self.labels, self.tree_types = (
             #   sample_data_from_labels_varying_tree(
              #      labels, self.rules, num_features, d_max
               # )
            #)
            self.features, self.labels, self.tree_types = (
                sample_data_from_labels_varying_tree_tensorized_d_values(
                    labels, self.rules,create_probabilities(m_2, m_3, num_layers), num_features, d_max, m_2
                )
            )
        elif return_type and d_5_4_set==0 and top_ter==0 and eta_set==1:
            #self.features, self.labels, self.tree_types = (
             #   sample_data_from_labels_varying_tree(
              #      labels, self.rules, num_features, d_max
               # )
            #)
            self.features, self.labels, self.tree_types = (
                sample_data_from_labels_varying_tree_tensorized_d_values(
                    labels, self.rules,create_probabilities_eta(m_2, m_3, num_layers,eta), num_features, d_max, m_2
                )
            )

        elif return_type and d_5_4_set:
            self.features,self.labels,self.tree_types=sample_data_from_labels_varying_tree_d_5_4(labels, self.rules, num_features, d_max)
        elif top_ter and return_type==1:
            self.features, self.labels, self.tree_types  = sample_data_from_labels_varying_tree_top_ter(
                labels, self.rules, num_features, d_max
            )
        else:
            self.features, self.labels = (
                sample_data_from_labels_varying_tree_tensorized(
                    labels,
                    self.rules,
                    create_probabilities(m_2, m_3, num_layers),
                    num_features,
                    d_max,
                )
            )

        if "onehot" not in input_format:
            assert not whitening, "Whitening only implemented for one-hot encoding"

        if "onehot" in input_format:

            self.features = F.one_hot(
                self.features.long(),
                num_classes=num_features + 1,  # Including the extra feature for padding
            ).float()

            self.features = self.features.permute(
                0, 2, 1
            )  # Shape: (batch_size, num_features+1, input_size)

            # Create a mask for positions with the fake token
            mask = self.features[:, -1, :] == 1  # Shape: (batch_size, input_size)

            # Remove the extra feature used for padding
            self.features = self.features[
                :, :-1, :
            ]  # Shape: (batch_size, num_features, input_size)

            if whitening:
                inv_sqrt_norm = (1.0 - 1.0 / num_features) ** -0.5
                self.features = (self.features - 1.0 / num_features) * inv_sqrt_norm

            # Apply the mask to set elements to zero
            self.features[mask.unsqueeze(1).expand_as(self.features)] = 0
            batch_size, num_features, input_size = self.features.shape
            print("done")
            # print(sum_of_squares)

            if padding_tail:
                if num_layers==2:
                    pad_size = 1
                    pad_tensor = torch.zeros(
                        batch_size,
                        num_features,
                        pad_size,
                        device=self.features.device,
                        dtype=self.features.dtype,
                    )
                    self.features = torch.cat((self.features, pad_tensor), dim=2)
                elif num_layers==3:
                    pad_size = 3
                    pad_tensor = torch.zeros(
                        batch_size,
                        num_features,
                        pad_size,
                        device=self.features.device,
                        dtype=self.features.dtype,
                    )
                    self.features = torch.cat((self.features, pad_tensor), dim=2)

        elif "long" in input_format:
            self.features = self.features.long() + 1

        else:
            raise ValueError

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
                idx: sample index

        Returns:
            Feature-label pairs at index
        """
        x, y = self.features[idx], self.labels[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y


class MixedRandomHierarchyModel(Dataset):
    """
    Implement the Random Hierarchy Model (RHM) as a PyTorch dataset.
    """

    def __init__(
        self,
        num_features=8,  # vocabulary size
        num_classes=2,  # number of classes
        fraction_rules=0.5,  # number of synonymic low-level representations (multiplicity)
        rule_sequence_type=2,
        s_2=2,
        s_3=3,  # size of the low-level representations
        num_layers=2,
        max_data=1000,  # number of levels in the hierarchy
        seed_rules=0,
        seed_sample=1,
        train_size=-1,
        test_size=0,
        input_format="onehot",
        replacement=False,
        whitening=0,
        padding=0,
        padding_central=0,
        padding_tail=0,
        d_5_set=0,
        d_5_single=0,
        non_overlapping=0,
        check_overlap=0,
        transform=None,
    ):

        v = num_features
        f = fraction_rules/v
        m_2 = int(f * v)
        m_3 = int(f * v**2)
        self.num_features = num_features
        self.m_2 = m_2
        self.m_3 = m_3
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.s_2 = s_2
        self.s_3 = s_3
        self.fraction_rules = fraction_rules
        self.rule_sequence_type = rule_sequence_type
        if d_5_set==1:
            self.max_data = 2*v*m_2**2*m_3
        elif d_5_single==1:
            self.max_data = v*m_2**2*m_3
        else:  
            self.max_data = max_data
        if non_overlapping==0:
            self.rules = sample_mixed_rules(
                num_features, num_classes, m_2, m_3, s_2, s_3, num_layers, seed_rules
            )
        else:
            self.rules = sample_non_overlapping_mixed_rules(
                num_features, num_classes, m_2, m_3, s_2, s_3, num_layers, seed_rules
            )
        max_rule_types = int(np.floor((3**num_layers - 1) / 2))

        if self.rule_sequence_type == 1:
            rule_types = [i % 2 for i in range(max_rule_types)]
        elif self.rule_sequence_type == 2:
            rule_types = [(i + 1) % 2 for i in range(max_rule_types)]
        elif self.rule_sequence_type == 3:
            rule_types = [0, 0, 0]
        elif self.rule_sequence_type == 4:
            rule_types = [0, 1, 1]
        elif self.rule_sequence_type == 5:
            rule_types = [1, 0, 0, 0]
        elif self.rule_sequence_type == 6:
            rule_types = [1, 1, 0, 1]
        elif self.rule_sequence_type == 7:
            rule_types = [1, 1, 1, 1]

        # tree_structure,input_size,max_data=reconstruct_tree_structure(rule_types,num_classes,m_2,m_3,num_layers)

        if not replacement and d_5_set==0:
            if train_size == -1:
                samples = torch.arange(self.max_data)

            else:
                # test_size = min( test_size, max_data-train_size)
                random.seed(seed_sample)
                print(self.max_data)
                print(train_size + test_size)
                samples = torch.tensor(
                    random.sample(range(self.max_data), train_size + test_size)
                )
            print(self.rule_sequence_type)
            self.features, self.labels = sample_data_from_indices_fixed_tree(
                samples,
                self.rules,
                rule_types,
                num_classes,
                m_2,
                m_3,
                self.rule_sequence_type,
            )

        if not replacement and d_5_set:
            if train_size == -1:
                samples = torch.arange(self.max_data)

            else:
                # test_size = min( test_size, max_data-train_size)
                random.seed(seed_sample)
                print(self.max_data)
                print(train_size + test_size)
                samples = torch.tensor(
                    random.sample(range(self.max_data), train_size + test_size)
                )
            print(self.rule_sequence_type)

            self.features, self.labels = sample_data_from_indices_d_5(
                samples,
                self.rules,
                num_classes,
                m_2,
                m_3
            )
        elif replacement and d_5_set and check_overlap==0:
            samples = torch.randint(0, self.max_data, (train_size + test_size,))
            self.features, self.labels = sample_data_from_indices_d_5(
                samples,
                self.rules,
                num_classes,
                m_2,
                m_3 
            )
        elif replacement and d_5_set and check_overlap==1:
            samples = torch.randint(0, self.max_data, (train_size + test_size,))
            self.features, self.labels,self.overlap_flags = sample_data_from_indices_d_5_check_overlap(
                samples,
                self.rules,
                num_classes,
                m_2,
                m_3 
            )
            print("overlap flags")

        else:
            torch.manual_seed(seed_sample)
            if train_size == -1:
                labels = torch.randint(
                    low=0, high=num_classes, size=(self.max_data + test_size,)
                )
            else:
                labels = torch.randint(
                    low=0, high=num_classes, size=(train_size + test_size,)
                )
            self.features, self.labels = sample_data_from_labels_fixed_tree(
                labels, self.rules, rule_types
            )

        if "onehot" not in input_format:
            assert not whitening, "Whitening only implemented for one-hot encoding"

        if "onehot" in input_format:

            self.features = F.one_hot(
                self.features.long(), num_classes=num_features
            ).float()

            if whitening:

                inv_sqrt_norm = (1.0 - 1.0 / num_features) ** -0.5
                self.features = (self.features - 1.0 / num_features) * inv_sqrt_norm
            self.features = self.features.permute(0, 2, 1)
            batch_size, num_features, input_size = self.features.shape
            print(input_size)

            if padding:

                target_size = 9
                if input_size < target_size:
                    pad_size = target_size - input_size
                    pad_tensor = torch.zeros(
                        batch_size,
                        num_features,
                        pad_size,
                        device=self.features.device,
                        dtype=self.features.dtype,
                    )
                    self.features = torch.cat(
                        [self.features, pad_tensor], dim=2
                    )  # Stack zeros at the end along the last dimension
                sum_of_squares = torch.sum(self.features**2, dim=(1, 2), keepdim=True)
                sum_of_squares = torch.round(sum_of_squares).to(torch.int)
                sum_of_squares = sum_of_squares.squeeze()
                print(sum_of_squares)
            if padding_central:
                target_size = 10
                if input_size < target_size:
                    pad_size = target_size - input_size
                    left_pad = pad_size // 2
                    right_pad = (
                        pad_size - left_pad
                    )  # Ensures right side gets extra if pad_size is odd

                    left_pad_tensor = torch.zeros(
                        batch_size,
                        num_features,
                        left_pad,
                        device=self.features.device,
                        dtype=self.features.dtype,
                    )
                    right_pad_tensor = torch.zeros(
                        batch_size,
                        num_features,
                        right_pad,
                        device=self.features.device,
                        dtype=self.features.dtype,
                    )

                    self.features = torch.cat(
                        [left_pad_tensor, self.features, right_pad_tensor], dim=2
                    )  # Stack zeros at the beginning and end along the last dimension
            if padding_tail:

                target_size = 10
                if input_size < target_size:
                    pad_size = target_size - input_size
                    pad_tensor = torch.zeros(
                        batch_size,
                        num_features,
                        pad_size,
                        device=self.features.device,
                        dtype=self.features.dtype,
                    )
                    self.features = torch.cat(
                        [self.features, pad_tensor], dim=2
                    )  # Stack zeros at the end along the last dimension
        elif "long" in input_format:
            self.features = self.features.long() + 1

        else:
            raise ValueError

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
                idx: sample index

        Returns:
            Feature-label pairs at index
        """
        x, y = self.features[idx], self.labels[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y


class RandomHierarchyModel(Dataset):
    """
    Implement the Random Hierarchy Model (RHM) as a PyTorch dataset.
    """

    def __init__(
        self,
        num_features=8,  # vocavulary size
        num_classes=2,  # number of classes
        num_synonyms=2,  # number of synonymic low-level representations (multiplicity)
        tuple_size=2,  # size of the low-level representations
        num_layers=2,  # number of levels in the hierarchy
        probability=None,  # for assigning nonuniform probabilities to production rules
        seed_rules=0,
        seed_sample=1,
        train_size=-1,
        test_size=0,
        input_format="onehot",
        whitening=0,
        transform=None,
        replacement=False,
        bonus={},
    ):

        self.num_features = num_features
        self.num_synonyms = num_synonyms
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.tuple_size = tuple_size

        self.rules = sample_rules(
            num_features,
            num_classes,
            num_synonyms,
            tuple_size,
            num_layers,
            seed=seed_rules,
        )

        max_data = num_classes * num_synonyms ** (
            (tuple_size**num_layers - 1) // (tuple_size - 1)
        )
        assert train_size >= -1, "train_size must be greater than or equal to -1"

        if max_data > sys.maxsize and not replacement:
            print(
                "Max dataset size cannot be represented with int64! Using sampling with replacement."
            )
            warnings.warn(
                "Max dataset size cannot be represented with int64! Using sampling with replacement.",
                RuntimeWarning,
            )
            replacement = True

        if not replacement:

            assert (
                probability is None
            ), "nonuniform probability only implemented for sampling with replacement."
            if train_size == -1:
                samples = torch.arange(max_data)

            else:
                test_size = min(test_size, max_data - train_size)
                random.seed(seed_sample)
                samples = torch.tensor(
                    random.sample(range(max_data), train_size + test_size)
                )

            self.features, self.labels = sample_data_from_indices(
                samples,
                self.rules,
                num_features,
                num_classes,
                num_synonyms,
                tuple_size,
                num_layers,
                bonus,
            )

        else:

            assert (
                not bonus
            ), "bonus data only implemented for sampling without replacement"
            # TODO: implement bonus data for sampling with replacement
            torch.manual_seed(seed_sample)
            if train_size == -1:
                labels = torch.randint(
                    low=0, high=num_classes, size=(max_data + test_size,)
                )
            else:
                labels = torch.randint(
                    low=0, high=num_classes, size=(train_size + test_size,)
                )
            if probability is None:
                self.features, self.labels = sample_data_from_labels_unif(
                    labels, self.rules
                )
            else:
                self.probability = probability
                self.features, self.labels = sample_data_from_labels(
                    labels, self.rules, self.probability
                )

        if "onehot" not in input_format:
            assert not whitening, "Whitening only implemented for one-hot encoding"

        if "tuples" in input_format:
            self.features = base2dec(
                self.features.view(self.features.size(0), -1, tuple_size), num_features
            )
            if bonus:
                if "synonyms" in bonus:
                    for k in bonus["synonyms"].keys():
                        bonus["synonyms"][k] = base2dec(
                            bonus["synonyms"][k].view(
                                bonus["synonyms"][k].size(0), -1, tuple_size
                            ),
                            num_features,
                        )

                if "noise" in bonus:
                    for k in bonus["noise"].keys():
                        bonus["noise"][k] = base2dec(
                            bonus["noise"][k].view(
                                bonus["synonyms"][k].size(0), -1, tuple_size
                            ),
                            num_features,
                        )

        if "onehot" in input_format:

            self.features = F.one_hot(
                self.features.long(),
                num_classes=(
                    num_features
                    if "tuples" not in input_format
                    else num_features**tuple_size
                ),
            ).float()
            if bonus:
                if "synonyms" in bonus:
                    for k in bonus["synonyms"].keys():
                        bonus["synonyms"][k] = F.one_hot(
                            bonus["synonyms"][k].long(),
                            num_classes=(
                                num_features
                                if "tuples" not in input_format
                                else num_features**tuple_size
                            ),
                        ).float()
                        bonus["synonyms"][k] = bonus["synonyms"][k].permute(0, 2, 1)
                if "noise" in bonus:
                    for k in bonus["noise"].keys():
                        bonus["noise"][k] = F.one_hot(
                            bonus["noise"][k].long(),
                            num_classes=(
                                num_features
                                if "tuples" not in input_format
                                else num_features**tuple_size
                            ),
                        ).float()
                        bonus["noise"][k] = bonus["noise"][k].permute(0, 2, 1)

            if whitening:

                inv_sqrt_norm = (1.0 - 1.0 / num_features) ** -0.5
                self.features = (self.features - 1.0 / num_features) * inv_sqrt_norm
                if bonus:
                    if "synonyms" in bonus:
                        for k in bonus["synonyms"].keys():
                            bonus["synonyms"][k] = (
                                bonus["synonyms"][k] - 1.0 / num_features
                            ) * inv_sqrt_norm

                    if "noise" in bonus:
                        for k in bonus["noise"].keys():
                            bonus["noise"][k] = (
                                bonus["noise"][k] - 1.0 / num_features
                            ) * inv_sqrt_norm

            self.features = self.features.permute(0, 2, 1)

        elif "long" in input_format:
            self.features = self.features.long() + 1

            if bonus:
                if "synonyms" in bonus:
                    for k in bonus["synonyms"].keys():
                        bonus["synonyms"][k] = bonus["synonyms"][k].long() + 1

                if "noise" in bonus:
                    for k in bonus["noise"].keys():
                        bonus["noise"][k] = bonus["noise"][k].long() + 1

        else:
            raise ValueError

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
                idx: sample index

        Returns:
            Feature-label pairs at index
        """
        x, y = self.features[idx], self.labels[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y
