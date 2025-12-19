#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 22:51:44 2025

@author: Rondini Davide
"""

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- CONFIG ---
CLASSES_FILE = "classes.txt"
OUTPUT_FILE = "class_embeddings.pt"
MODEL_NAME = 'bert-base-uncased'

def load_class_names(path):
    id2name = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                id2name[int(parts[0])] = parts[1]
    return id2name

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating Class Embeddings (Mean Pooled) on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    id2name = load_class_names(CLASSES_FILE)
    sorted_ids = sorted(id2name.keys())

    embeddings = []

    print(f"Encoding {len(sorted_ids)} classes...")
    with torch.no_grad():
        for cid in tqdm(sorted_ids):
            text = id2name[cid]
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs)

            # --- Mean Pooling ---
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask'].unsqueeze(-1)

            # Sum embeddings (masking padding)
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            # Count active tokens
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)

            # Average
            mean_pooled = sum_embeddings / sum_mask
            embeddings.append(mean_pooled.cpu())

    # Concatenate to [Num_Classes, 768]
    final_tensor = torch.cat(embeddings, dim=0)
    torch.save(final_tensor, OUTPUT_FILE)
    print(f"Saved embeddings to {OUTPUT_FILE} shape: {final_tensor.shape}")

if __name__ == "__main__":
    main()