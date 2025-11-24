#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 18:11:53 2025

@author: daviderondini
"""

import torch
from torch.utils.data import Dataset
import json

class ReviewDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len, num_classes):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_classes = num_classes
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item['text']
        labels_indices = item['labels']

        # Text tokenization
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Multi-hot encoding
        labels_tensor = torch.zeros(self.num_classes, dtype=torch.float)
        for label_idx in labels_indices:
            if label_idx < self.num_classes:
                labels_tensor[label_idx] = 1.0

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels_tensor
        }