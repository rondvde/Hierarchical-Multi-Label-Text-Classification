#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 16:24:02 2025

@author: daviderondini
"""

import json
import re
import networkx as nx
from tqdm import tqdm

# --- CONFIGURATION ---
FILES = {
    "test_text": "test/test_corpus.txt",
    "classes": "classes.txt",
    "hierarchy": "class_hierarchy.txt",
    "keywords": "class_related_keywords.txt"
}
OUTPUT_FILE = "test/val_data.json"

def load_data():
    print("--- LOADING BASIC KNOWLEDGE ---")
    
    # 1. ID <-> Name
    name2id = {}
    with open(FILES["classes"], 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                name2id[parts[1]] = parts[0]

    # 2. HIERARCHY
    G = nx.DiGraph()
    with open(FILES["hierarchy"], 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                G.add_edge(parts[0], parts[1])

    # 3. KEYWORDS
    class_keywords = {} 
    with open(FILES["keywords"], 'r') as f:
        for line in f:
            if ':' in line:
                name_part, kw_part = line.strip().split(':', 1)
                if name_part in name2id:
                    cid = name2id[name_part]
                    kws = [k.strip().lower() for k in kw_part.split(',')]
                    class_keywords[cid] = kws
    
    print(f"Keywords loaded for {len(class_keywords)} classes.")
    return name2id, G, class_keywords

def propagate_labels(labels, G):
    expanded_labels = set(labels)
    for label in labels:
        if label in G:
            ancestors = nx.ancestors(G, label)
            expanded_labels.update(ancestors)
    return list(expanded_labels)

def main():
    name2id, G, class_keywords = load_data()
    
    val_dataset = []
    
    print("\n--- GENERATION SILVER LABELS FOR VALIDATION ---")
    
    try:
        with open(FILES["test_text"], 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"ERROR: File not found {FILES['test_text']}")
        return
        
    for line in tqdm(lines):
        parts = line.strip().split('\t', 1)
        if len(parts) < 2: continue
        
        rid, text_original = parts[0], parts[1]
        
        # 1. Text cleaning
        text_lower = text_original.lower()
        text_clean = re.sub(r'[^a-z0-9\s]', ' ', text_lower)
        
        found_labels = []
        
        # 2. MATCHING ALGORITHM
        for cid, kws in class_keywords.items():
            for kw in kws:
                if kw in text_clean:
                    pattern = r'\b' + re.escape(kw) + r'\b'
                    if re.search(pattern, text_clean):
                        found_labels.append(cid)
                        break
        
        # 3. If label found, propagate it and save it
        if found_labels:
            final_labels = propagate_labels(found_labels, G)
            
            final_labels_int = [int(x) for x in final_labels]
            
            val_dataset.append({
                "text": text_original,
                "labels": final_labels_int
            })

    # Saving
    print(f"\nProcessed and saved reviews: {len(val_dataset)}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(val_dataset, f)
        
    print(f"File saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()