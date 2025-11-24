#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 19:27:13 2025

@author: daviderondini
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import json
import numpy as np
from tqdm import tqdm
import os

# --- CONFIGURATION ---
MODEL_PATH = "saved_models/best_model_silver.pt"
RAW_DATA_PATH = "train/train_corpus_cleaned.txt" 
OUTPUT_JSON = "to_annotate.json"
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 32
NUM_CLASSES = 531
TOP_K = 1000

# --- 1. DATASET FOR INFERENCE ---
class InferenceDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) >= 2:
                        self.data.append((parts[0], parts[1]))
        except FileNotFoundError:
            print(f"File not found.")
            exit()
            
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rid, text = self.data[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'id': rid,
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

# --- 2. SETUP ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    print("Using CPU")

print(f"Device selected: {device}")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
dataset = InferenceDataset(RAW_DATA_PATH, tokenizer, MAX_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Loading model...")
try:
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_CLASSES, 
        problem_type="multi_label_classification"
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
except FileNotFoundError:
    print(f"Model not found.")
    exit()

# --- 3. EVALUATING UNCERTAINTY ---
results = []

print(f"--- Evualuating uncertainty on {len(dataset)} reviews ---")
with torch.no_grad():
    for batch in tqdm(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ids = batch['id']
        texts = batch['text']
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits) 
        
        probs = probs.clamp(1e-9, 1 - 1e-9)
        
        entropy_per_class = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs))
        
        # Sum of TOP 5 uncertainty per review.
        top_k_entropy, _ = torch.topk(entropy_per_class, k=5, dim=1)
        uncertainty_score = top_k_entropy.mean(dim=1).cpu().numpy()
        
        for i in range(len(ids)):
            results.append({
                "id": ids[i],
                "text": texts[i],
                "uncertainty": float(uncertainty_score[i])
            })

# --- 4. SELECTION AND SAVING ---
print("Ordering by descending uncertainty...")
results.sort(key=lambda x: x['uncertainty'], reverse=True)

top_uncertain = results[:TOP_K]

print(f"Top 3 uncertainty (Max): {[f'{x['uncertainty']:.4f}' for x in top_uncertain[:3]]}")
print(f"Bottom 3 uncertainty (Min): {[f'{x['uncertainty']:.4f}' for x in top_uncertain[-3:]]}")

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(top_uncertain, f, indent=4)

print(f"{len(top_uncertain)} saved reviews in {OUTPUT_JSON}")