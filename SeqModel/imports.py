import os
import csv
import pandas as pd
import time
import pickle
import sys
import datetime
import numpy as np

# train the model
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import wandb
import torch.nn as nn

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1)
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import random

def augment_sequence(sequences, max_changes, action_prob , time_prob):
    all_sequences = []
    for sequence in sequences:
        all_sequences.append(sequence)
       

        augmented_sequence = []
        number_of_changes = 0
        i = 0

        while i <len(sequence):

            if number_of_changes>=max_changes:
                augmented_sequence.extend(sequence[i:])
                all_sequences.append(augmented_sequence)
                augmented_sequence = sequence[:i]
            
            token = sequence[i]
            if token.startswith('Q'): # dont change the problem id
                augmented_sequence.append(token)
                i+=1
                continue
            elif token.startswith('T'): # vary time intervals
                # Extract the time value and apply variance
                # 'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'TMAX',
                if random.random() < time_prob:
                    number_of_changes += 1
                    time_val = int(token[1:]) if token != 'TMAX' else 8
                    # choose a random time interval with normal distribution mean = time_val, std = time_prob
                    time_val = max(0, min(8, int(np.random.normal(time_val, time_prob))))
                    augmented_sequence.append('T' + str(time_val))
                else:
                    augmented_sequence.append(token)
            else: # 'UA','FA','PA','FH', 'UH', 'RH','FE', 'UE','RF', 'RC', 'B', 'C','M', 'S',
            # Randomly decide to repeat or skip an action
                action = token
                p = random.random()
                number_of_changes += 1
                if p < action_prob:
                    # Repeat action
                    # randomply select a time interval from T0, T1, T2
                    augmented_sequence.extend([action, 'T' + str(random.randint(0, 2)), action])
                elif p < 2 * action_prob:
                    # Skip action
                    i+=2 # skip the time
                    continue
                elif p < 2.5 * action_prob:
                    # Insert a random action
                    action_list = ['UA','FA','PA','FH', 'UH', 'RH','FE', 'UE','RF', 'RC', 'B', 'C','M', 'S']
                    augmented_sequence.extend([random.choice(action_list), 'T' + str(random.randint(0, 2)), action])
                else:
                    augmented_sequence.append(action)
                    number_of_changes -= 1
            i+=1

        all_sequences.append(augmented_sequence)
    return all_sequences