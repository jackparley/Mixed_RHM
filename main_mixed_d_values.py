import os
import sys
import time
import copy

import numpy as np
import math
import random

import functools
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import pickle

import datasets
import models
import init
import measures
import collections
from collections import defaultdict

def run( args):

    # reduce batch_size when larger than train_size
    if (args.batch_size >= args.train_size):
        args.batch_size = args.train_size
   
    assert (args.train_size%args.batch_size)==0, 'batch_size must divide train_size!'
    args.num_batches = args.train_size//args.batch_size
    args.max_iters = args.max_epochs*args.num_batches

    train_loader, test_loader_indexed,data_type= init.init_data_mixed( args)


    model = init.init_model_mixed( args)
    model0 = copy.deepcopy( model)

    if args.scheduler_time is None:
        args.scheduler_time = args.max_iters
    criterion, optimizer, scheduler = init.init_training( model, args)
 
    print_ckpts, save_ckpts = init.init_loglinckpt( args.print_freq, args.max_iters, freq=args.save_freq,log_linear_switch=args.log_linear_switch)
    print_ckpt = next(print_ckpts)
    save_ckpt = next(save_ckpts)

    start_time = time.time()
    step = 0
    dynamics, best = init.init_output_mixed( model, criterion, train_loader, test_loader_indexed, args)
    #if args.checkpoints:

     #   output = {
      #      'model': copy.deepcopy(model.state_dict()),
       #     'state': dynamics[-1],
        #    'step': step
        #}
        #with open(args.outname+f'_t{0}', "wb") as handle:
         #   pickle.dump(args, handle)
          #  pickle.dump(output, handle)

   
    window_size = 5
    test_loss_window = collections.deque(maxlen=window_size)
    step_window = collections.deque(maxlen=window_size)
    stop_training = False  # Flag to signal when to stop training

    for epoch in range(args.max_epochs):

        model.train()
        optimizer.zero_grad()
        running_loss = 0.

        for batch_idx, (inputs, targets) in enumerate(train_loader):

            outputs = model(inputs.to(args.device))
            loss = criterion(outputs, targets.to(args.device))
            running_loss += loss.item()
            loss /= args.accumulation
            loss.backward()

            if ((batch_idx+1)%args.accumulation==0):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

                if step==print_ckpt:

                    #test_loss, test_acc = measures.test(model, test_loader, args.device)

                    # Initialize storage for each type
                    # type_losses = defaultdict(list)
                    # type_correct = defaultdict(int)
                    # type_counts = defaultdict(int)
                    # with torch.no_grad():
                    #     for inputs, labels, indices in test_loader_indexed:
                    #         inputs = inputs.to(args.device)
                    #         labels = labels.to(args.device)
                    #        
                    #         outputs = model(inputs)
                    #        
                    #         # Compute loss for the batch (element-wise)
                    #         losses = F.cross_entropy(outputs, labels, reduction='none')
                    #        
                    #         # Predicted classes
                    #         preds = outputs.argmax(dim=1)
                    #         #sum_of_squares = torch.sum(inputs**2, dim=(1, 2), keepdim=True)
                    #         #sum_of_squares = torch.round(sum_of_squares).to(torch.int)
                    #         #sum_of_squares = sum_of_squares.squeeze()-4  #Indices from 0 to 8

                    #        
                    #         for i in range(len(labels)):
                    #             #type_id = sum_of_squares[i].item()  # Lookup type (0 to 6)
                    #             #print("type_id_old:", type_id)
                    #             type_id=data_type[indices[i]]
                    #             #print("type_id_new:", type_id)
                    #             type_losses[type_id].append(losses[i].item())
                    #             type_correct[type_id] += (preds[i] == labels[i]).item()
                    #             type_counts[type_id] += 1

                    # # Compute metrics per type
                    # counts_sum=0
                    # loss_sum = 0
                    # correct_sum = 0
                    # avg_losses=np.zeros(7)
                    # avg_accs=np.zeros(7)
                    # for type_id in range(7):
                    #     if type_counts[type_id] == 0:
                    #         #print(f"Type {type_id}: No samples.")
                    #         continue
                    #     avg_loss = sum(type_losses[type_id]) / type_counts[type_id]
                    #     avg_losses[type_id]=avg_loss
                    #     accuracy = type_correct[type_id] / type_counts[type_id]
                    #     avg_accs[type_id]=accuracy
                    #     print(f"Type {type_id}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
                    #     counts_sum += type_counts[type_id]
                    #     loss_sum += sum(type_losses[type_id])
                    #     correct_sum += type_correct[type_id]
                    # print(f"Total: Loss = {loss_sum/counts_sum:.4f}, Accuracy = {correct_sum/counts_sum:.4f}")

                    # test_loss = loss_sum/counts_sum
                    # test_acc = correct_sum/counts_sum
                    # 


                    # print('step : ',step, '\t train loss: {:06.4f}'.format(running_loss/(batch_idx+1)), ',test loss: {:06.4f}'.format(test_loss))
                    print_ckpt = next(print_ckpts)

                    if step>=save_ckpt:

                        print(f'Checkpoint at step {step}, saving data ...')


                            # Initialize storage for each type
                        type_losses = defaultdict(list)
                        type_correct = defaultdict(int)
                        type_counts = defaultdict(int)
                        with torch.no_grad():
                            for inputs, labels, indices in test_loader_indexed:
                                inputs = inputs.to(args.device)
                                labels = labels.to(args.device)
                            
                                outputs = model(inputs)
                            
                                # Compute loss for the batch (element-wise)
                                losses = F.cross_entropy(outputs, labels, reduction='none')
                            
                                # Predicted classes
                                preds = outputs.argmax(dim=1)
                                #sum_of_squares = torch.sum(inputs**2, dim=(1, 2), keepdim=True)
                                #sum_of_squares = torch.round(sum_of_squares).to(torch.int)
                                #sum_of_squares = sum_of_squares.squeeze()-4  #Indices from 0 to 8

                            
                                for i in range(len(labels)):
                                    #type_id = sum_of_squares[i].item()  # Lookup type (0 to 6)
                                    #print("type_id_old:", type_id)
                                    type_id=data_type[indices[i]]
                                    #print("type_id_new:", type_id)
                                    type_losses[type_id].append(losses[i].item())
                                    type_correct[type_id] += (preds[i] == labels[i]).item()
                                    type_counts[type_id] += 1

                        # Compute metrics per type
                        counts_sum=0
                        loss_sum = 0
                        correct_sum = 0
                        avg_losses=np.zeros(7)
                        avg_accs=np.zeros(7)
                        for type_id in range(7):
                            if type_counts[type_id] == 0:
                                #print(f"Type {type_id}: No samples.")
                                continue
                            avg_loss = sum(type_losses[type_id]) / type_counts[type_id]
                            avg_losses[type_id]=avg_loss
                            accuracy = type_correct[type_id] / type_counts[type_id]
                            avg_accs[type_id]=accuracy
                            print(f"Type {type_id}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
                            counts_sum += type_counts[type_id]
                            loss_sum += sum(type_losses[type_id])
                            correct_sum += type_correct[type_id]
                        print(f"Total: Loss = {loss_sum/counts_sum:.4f}, Accuracy = {correct_sum/counts_sum:.4f}")

                        test_loss = loss_sum/counts_sum
                        test_acc = correct_sum/counts_sum
                    


                        print('step : ',step, '\t train loss: {:06.4f}'.format(running_loss/(batch_idx+1)), ',test loss: {:06.4f}'.format(test_loss))

                        train_loss, train_acc = measures.test(model, train_loader, args.device)
                        save_dict = {'t': step, 'trainloss': train_loss, 'trainacc': train_acc, 'testloss': test_loss, 'testacc': test_acc, 'avg_losses': avg_losses, 'avg_accs': avg_accs}
                        dynamics.append(save_dict)

                        end_time = time.time()
                        print('Training time: ', end_time-start_time)
                        training_time = end_time - start_time
                        output_data = {
                            'dynamics': dynamics,
                            'training_time': training_time
                        }

                        with open(args.outname, 'wb') as file:
                            pickle.dump(output_data, file)

                         # Store (1 - test_acc) in the deque
                        test_loss_window.append(test_loss)
                        step_window.append(step)

                        if step > 1e2:
                            max_val = max(test_loss_window)
                            min_val = min(test_loss_window)
                            min_step = min(step_window)
                            variation = (max_val - min_val) / max_val if max_val != 0 else 0
                            # Compute normalized fluctuation (Coefficient of Variation)
                            print(f"Variation: {variation}")
                            var_step = step - min_step

                            # Compute the slope of the test loss trend
                            if len(test_loss_window) ==window_size:  # Ensure enough points for trend detection
                                steps_array = np.array(list(step_window))
                                losses_array = np.array(list(test_loss_window))

                                # Fit a linear regression line
                                slope, _ = np.polyfit(steps_array, losses_array, 1)  

                                # Check the consistency of the increase
                                increasing_steps = sum(
                                    losses_array[i] > losses_array[i - 1] for i in range(1, len(losses_array))
                                )
                                consistency_ratio = increasing_steps / len(losses_array)

                                print(f"Test loss slope: {slope}, Consistency Ratio: {consistency_ratio}")
                                    # Condition to detect a noisy plateau
                                if args.check_plateau==1 and args.stopping_criteria==1:
                                    if (variation < 0.12 and var_step > 5000 and step > 5e4):  
                                        # Stop if plateauing or consistently increasing (more than 70% of the time)
                                        print("Training stopped: Loss plateau or consistently increasing trend detected.")
                                        train_loss, train_acc = measures.test(model, train_loader, args.device)
                                        save_dict = {'t': step, 'trainloss': train_loss, 'trainacc': train_acc, 'testloss': test_loss, 'testacc': test_acc, 'avg_losses': avg_losses, 'avg_accs': avg_accs}
                                        dynamics.append(save_dict)
                                        stop_training = True  # Set flag to stop both loops
                                        break  # Stop training
                                    elif (slope > 0 and consistency_ratio > 0.2):
                                        # Stop if plateauing or consistently increasing (more than 70% of the time)
                                        print("Training stopped: consistently increasing trend detected.")
                                        train_loss, train_acc = measures.test(model, train_loader, args.device)
                                        save_dict = {'t': step, 'trainloss': train_loss, 'trainacc': train_acc, 'testloss': test_loss, 'testacc': test_acc, 'avg_losses': avg_losses, 'avg_accs': avg_accs}
                                        dynamics.append(save_dict)
                                        stop_training = True  # Set flag to stop both loops
                                        break  # Stop training
                                elif args.stopping_criteria==1 and args.check_plateau==0:
                                    if (slope > 0 and consistency_ratio > 0.5):  
                                        # Stop if plateauing or consistently increasing (more than 70% of the time)
                                        print("Training stopped: Loss plateau or consistently increasing trend detected.")
                                        train_loss, train_acc = measures.test(model, train_loader, args.device)
                                        save_dict = {'t': step, 'trainloss': train_loss, 'trainacc': train_acc, 'testloss': test_loss, 'testacc': test_acc, 'avg_losses': avg_losses, 'avg_accs': avg_accs}
                                        dynamics.append(save_dict)
                                        stop_training = True  # Set flag to stop both loops
                                        break  # Stop training
                        #if args.checkpoints:
                         #   output = {
                          #      'model': copy.deepcopy(model.state_dict()),
                           #     'state': dynamics[-1],
                            #    'step': step
                            #}
                            #with open(args.outname+f'_t{step}', "wb") as handle:
                             #   pickle.dump(output, handle)
                        #else:
                         #   output = {
                          #      'init': model0.state_dict(),
                           #     'best': best,
                            #    'model': copy.deepcopy(model.state_dict()),
                             #   'dynamics': dynamics,
                              #  'step': step
                            #}
                            #with open(args.outname, "wb") as handle:
                             #   pickle.dump(args, handle)
                              #  pickle.dump(output, handle)
                        save_ckpt = next(save_ckpts)

        if stop_training:  # Break outer loop
            break

        if (running_loss/(batch_idx+1)) <= args.loss_threshold:
        

            train_loss, train_acc = measures.test(model, train_loader, args.device)

            type_losses = defaultdict(list)
            type_correct = defaultdict(int)
            type_counts = defaultdict(int)
            with torch.no_grad():
                for inputs, labels, indices in test_loader_indexed:
                    inputs = inputs.to(args.device)
                    labels = labels.to(args.device)
                
                    outputs = model(inputs)
                
                    # Compute loss for the batch (element-wise)
                    losses = F.cross_entropy(outputs, labels, reduction='none')
                
                    # Predicted classes
                    preds = outputs.argmax(dim=1)
                    #sum_of_squares = torch.sum(inputs**2, dim=(1, 2), keepdim=True)
                    #sum_of_squares = torch.round(sum_of_squares).to(torch.int)
                    #sum_of_squares = sum_of_squares.squeeze()-4  #Indices from 0 to 8

                
                    for i in range(len(labels)):
                        #type_id = sum_of_squares[i].item()  # Lookup type (0 to 6)
                        #print("type_id_old:", type_id)
                        type_id=data_type[indices[i]]
                        #print("type_id_new:", type_id)
                        type_losses[type_id].append(losses[i].item())
                        type_correct[type_id] += (preds[i] == labels[i]).item()
                        type_counts[type_id] += 1

            # Compute metrics per type
            counts_sum=0
            loss_sum = 0
            correct_sum = 0
            avg_losses=np.zeros(7)
            avg_accs=np.zeros(7)
            for type_id in range(7):
                if type_counts[type_id] == 0:
                    #print(f"Type {type_id}: No samples.")
                    continue
                avg_loss = sum(type_losses[type_id]) / type_counts[type_id]
                avg_losses[type_id]=avg_loss
                accuracy = type_correct[type_id] / type_counts[type_id]
                avg_accs[type_id]=accuracy
                print(f"Type {type_id}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
                counts_sum += type_counts[type_id]
                loss_sum += sum(type_losses[type_id])
                correct_sum += type_correct[type_id]
            print(f"Total: Loss = {loss_sum/counts_sum:.4f}, Accuracy = {correct_sum/counts_sum:.4f}")

            test_loss = loss_sum/counts_sum
            test_acc = correct_sum/counts_sum# Initialize storage for each type



            save_dict = {'t': step, 'trainloss': train_loss, 'trainacc': train_acc, 'testloss': test_loss, 'testacc': test_acc, 'avg_losses': avg_losses, 'avg_accs': avg_accs}
            dynamics.append(save_dict)

            #if args.checkpoints:
             #   output = {
              #      'model': copy.deepcopy(model.state_dict()),
               #     'state': dynamics[-1],
                #    'step': step
                #}
                #with open(args.outname+f'_t{step}', "wb") as handle:
                 #   pickle.dump(output, handle)
            #else:
             #   output = {
              #      'init': model0.state_dict(),
               #     'best': best,
                #    'model': copy.deepcopy(model.state_dict()),
                 #   'dynamics': dynamics,
                  #  'step': step
                #}
                #with open(args.outname, "wb") as handle:
                 #   pickle.dump(args, handle)
                  #  pickle.dump(output, handle)
            break
    #filename = 'dynamics.pkl'

# Write the list to a pickle file  
    end_time = time.time()
    print('Training time: ', end_time-start_time)
    training_time = end_time - start_time
    output_data = {
        'dynamics': dynamics,
        'training_time': training_time
    }

    with open(args.outname, 'wb') as file:
        pickle.dump(output_data, file)

torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description='Learning the Random Hierarchy Model with deep neural networks')
parser.add_argument("--device", type=str, default='cuda')
'''
DATASET ARGS
'''
parser.add_argument('--dataset', type=str)
parser.add_argument('--num_features', metavar='v', type=int, help='number of features')
parser.add_argument('--num_classes', metavar='n', type=int, help='number of classes')
parser.add_argument('--num_tokens', type=int, help='input length')
parser.add_argument('--fraction_rules', metavar='f', type=float, help='fraction of rules')
parser.add_argument('--rule_sequence_type', type=int, help='rule sequence type')
parser.add_argument('--num_layers', metavar='L', type=int, help='number of layers')
parser.add_argument("--seed_rules", type=int, help='seed for the dataset')
parser.add_argument('--train_size', metavar='Ptr', type=int, help='training set size')
parser.add_argument('--batch_size', metavar='B', type=int, help='batch size')
parser.add_argument('--max_data', type=int, help='maximum number of data points')

#parser.add_argument('--test_size', metavar='Pte', type=int, help='test set size')
parser.add_argument("--seed_sample", type=int, help='seed for the sampling of train and testset')
parser.add_argument('--input_format', type=str, default='onehot')
parser.add_argument('--whitening', type=int, default=0)
parser.add_argument('--padding', type=int, default=0)
parser.add_argument('--padding_tail', type=int, default=0)
parser.add_argument('--padding_central', type=int, default=0)
parser.add_argument('--return_type', type=int, default=0)
parser.add_argument('--non_overlapping', type=int, default=0)
parser.add_argument('--replacement', default=False, action='store_true')
parser.add_argument('--d_5_4_set',type=int, default=0)
parser.add_argument('--check_overlap',default=0, type=int)
parser.add_argument('--top_ter',default=0, type=int)
parser.add_argument('--eta_set',default=0, type=int)
parser.add_argument('--eta',default=8, type=int)





'''
ARCHITECTURE ARGS
'''
parser.add_argument('--model', type=str, help='architecture (fcn, hcnn,hcnn_mixed, hlcn, transformer_mla)')
parser.add_argument('--depth', type=int, help='depth of the network')
parser.add_argument('--final_dim', type=int, help='final dimension for Gen CNN',default=1)

parser.add_argument('--width', type=int, help='width of the network')
parser.add_argument('--width_2', type=int, help='width of the top layer')

parser.add_argument('--bias', default=False, action='store_true')
parser.add_argument("--seed_model", type=int, help='seed for model initialization')
parser.add_argument('--mlp_dim', type=int, default=256)
'''
       TRAINING ARGS
'''
parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--accumulation', type=int, default=1)
parser.add_argument('--momentum', type=float, default=0.0)
parser.add_argument('--scheduler', type=str, default=None)
parser.add_argument('--scheduler_time', type=int, default=None)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--check_plateau', type=int, default=1)
parser.add_argument('--stopping_criteria', type=int, default=1)
parser.add_argument('--log_linear_switch', type=int, default=int(1e5))
'''
OUTPUT ARGS
'''
parser.add_argument('--print_freq', type=int, help='frequency of prints', default=16)
parser.add_argument('--save_freq', type=int, help='frequency of saves', default=2)
parser.add_argument('--checkpoints', default=False, action='store_true')
parser.add_argument('--loss_threshold', type=float, default=1e-3)
parser.add_argument('--outname', type=str, required=True, help='path of the output file')

args = parser.parse_args()
run( args)
