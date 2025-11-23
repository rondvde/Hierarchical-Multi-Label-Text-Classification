#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 20:16:36 2025

@author: RONDINI DAVIDE
"""

import os
import json
import networkx as nx
from tqdm import tqdm

# --- FILE CONFIGURATION ---
FILES = {
    "classes": "classes.txt",
    "hierarchy": "class_hierarchy.txt",
    "keywords": "class_related_keywords.txt",
    "train": "train/train_corpus.txt"
}
OUTPUT_FILE = "silver_labels.json"

def load_data():
    print("--- LOADING DATA ---")
    
    # 1. ID <-> Name
    id2name = {}
    name2id = {}
    with open(FILES["classes"], 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                cid, cname = parts[0], parts[1]
                id2name[cid] = cname
                name2id[cname] = cid
    print(f"Loaded classes: {len(id2name)}")

    # 2. Constructing hierarchy
    G = nx.DiGraph()
    with open(FILES["hierarchy"], 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                parent, child = parts[0], parts[1]
                G.add_edge(parent, child)
    print(f"Hierarchy loaded: {G.number_of_edges()} relations")

    # 3. Loading keywords and conversion of names in ID
    class_keywords = {} # {class_id: [keyword1, keyword2]}
    with open(FILES["keywords"], 'r') as f:
        for line in f:
            # Format: "class_name:k1,k2,k3"
            if ':' in line:
                name_part, kw_part = line.strip().split(':', 1)
                if name_part in name2id:
                    cid = name2id[name_part]
                    # keyword cleaning: lowercase and trim
                    kws = [k.strip().lower() for k in kw_part.split(',')]
                    class_keywords[cid] = kws
    print(f"Keywords loaded for {len(class_keywords)} classes")
    
    return id2name, name2id, G, class_keywords

def propagate_labels(labels, G):
    """
    If we assign a child class, we must assign all of its ancestors 
    (parents, grandparents) up to the root
    """
    expanded_labels = set(labels)
    for label in labels:
        if label in G:
            ancestors = nx.ancestors(G, label)
            expanded_labels.update(ancestors)
    return list(expanded_labels)

def generate_labels():
    id2name, name2id, G, class_keywords = load_data()
    
    silver_data = {} # {review_id: [label_id_1, label_id_2...]}
    
    print("\n--- IGENERATING SILVER LABELS ---")
    
    # reading training data line by line
    with open(FILES["train"], 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in tqdm(lines):
        parts = line.strip().split('\t', 1)
        if len(parts) < 2: continue
        
        rid, text = parts[0], parts[1].lower() # lowercasing text
        
        found_labels = []
        
        # MATCHING ALGORITHM
        for cid, kws in class_keywords.items():
            for kw in kws:
                if kw in text: 
                    found_labels.append(cid)
                    break
        
        # HIERARCHICAL PROPAGATION
        if found_labels:
            final_labels = propagate_labels(found_labels, G)
            silver_data[rid] = final_labels
            
    # SAVING
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(silver_data, f)
        
    print(f"\nDONE! {len(silver_data)} reviews labelled out of {len(lines)}.")
    print(f"file saved as: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_labels()