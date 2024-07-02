# Description: This file contains functions for preparing the datasets for training and evaluation.
import pandas as pd
import torch
from transformers import AutoTokenizer
import numpy as np
import os
import pickle
import json
from sklearn.model_selection import train_test_split

def define_x_y(df, x_col, y_col):
    """
    Define the features and target variables.
    """
    X = df[x_col]
    y = df[y_col]

    return X, y

def tokenize_smiles(X, y, pretrained_model, max_length, csv_file_name):
    """
    Tokenize SMILES strings and save them to a JSON file.
    """
    # Load the tokenizer using AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    input_ids = []
    attention_masks = []
    
    # Tokenization
    for text in X.tolist():
        encoded = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_length, truncation=True,
                                        padding='max_length', return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded['input_ids'].tolist())
        attention_masks.append(encoded['attention_mask'].tolist())

    # Create a dictionary to store all the data
    data = {
        'input_ids': input_ids,
        'attention_masks': attention_masks,
        'labels': y.values.tolist()
    }

    # Create the output if it doesn't exist
    output_dir = "datasets/tokens"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the dictionary as a JSON file
    output_file_name = f"{output_dir}/{os.path.splitext(os.path.basename(csv_file_name))[0]}.json"
    with open(output_file_name, 'w') as f:
        json.dump(data, f, cls=NumpyArrayEncoder)

    # Return the dictionary (optional)
    return data

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def split_data(directory, train_size, val_size, test_size, random_state):
    """
    Iterate over a directory, read JSON files, and perform a split.
    """
    for file in os.listdir(directory):
        if file.endswith('json'):
            filename = os.fsdecode(file)
            filepath = os.path.join(directory, filename)

            # Load data from JSON file
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Split data into training, validation, and testing sets
            input_ids = np.asarray(data['input_ids'])
            attention_masks = np.asarray(data['attention_masks'])
            labels = np.asarray(data['labels'])

            input_ids_train, input_ids_val_test, attention_masks_train, attention_masks_val_test, labels_train, labels_val_test = train_test_split(
                input_ids, attention_masks, labels, test_size=val_size + test_size, train_size=train_size, random_state=random_state, stratify=labels
            )

            input_ids_val, input_ids_test, attention_masks_val, attention_masks_test, labels_val, labels_test = train_test_split(
                input_ids_val_test, attention_masks_val_test, labels_val_test, test_size=test_size / (val_size + test_size), random_state=random_state, stratify=labels_val_test
            )

            # Save the split data to JSON files
            with open(os.path.join("datasets/train", f'train_{filename}'), 'w', encoding='utf-8') as f:
                json.dump({
                    'input_ids': input_ids_train.tolist(),
                    'attention_masks': attention_masks_train.tolist(),
                    'labels': labels_train.tolist()
                }, f, cls=NumpyArrayEncoder)

            with open(os.path.join("datasets/val", f'val_{filename}'), 'w', encoding='utf-8') as f:
                json.dump({
                    'input_ids': input_ids_val.tolist(),
                    'attention_masks': attention_masks_val.tolist(),
                    'labels': labels_val.tolist()
                }, f, cls=NumpyArrayEncoder)

            with open(os.path.join("datasets/test", f'test_{filename}'), 'w', encoding='utf-8') as f:
                json.dump({
                    'input_ids': input_ids_test.tolist(),
                    'attention_masks': attention_masks_test.tolist(),
                    'labels': labels_test.tolist()
                }, f, cls=NumpyArrayEncoder)