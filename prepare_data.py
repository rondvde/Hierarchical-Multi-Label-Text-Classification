#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 18:11:53 2025

@author: daviderondini
"""

import json
import random

# --- CONFIGURATION ---
FILES = {
    "train_text": "train/train_corpus_cleaned.txt", 
    "silver_labels": "silver_labels.json",
    "classes": "classes.txt"
}

OUTPUT_TRAIN = "train/train_data.json"

def main():
    print("--- DATA FORMATTING ---")
    
    # 1. Loading silver labels
    print(f"Reading {FILES['silver_labels']}...")
    try:
        with open(FILES["silver_labels"], 'r') as f:
            silver_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File {FILES['silver_labels']} not found.")
        return
        
    print(f"{len(silver_data)} labels found.")

    # 2. Merging text + labels
    print(f"Reading {FILES['train_text']} and merging...")
    dataset = []
    
    try:
        with open(FILES["train_text"], 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"ERROR: File {FILES['train_text']} not found.")
        return

    for line in lines:
        parts = line.strip().split('\t', 1)
        if len(parts) < 2: continue
        
        rid = parts[0]
        text = parts[1].strip()
        
        if rid in silver_data:
            labels_list = silver_data[rid]
            labels_int = [int(x) for x in labels_list]
            
            dataset.append({
                "text": text,
                "labels": labels_int
            })

    print(f"Created {len(dataset)} total samples for the training.")

    # 3. Saving
    with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
        json.dump(dataset, f)
        
    print(f"\n FILE SAVED: {OUTPUT_TRAIN}")

if __name__ == "__main__":
    main()