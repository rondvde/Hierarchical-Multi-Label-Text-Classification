#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 23:17:41 2025

@author: daviderondini
"""

import json
import networkx as nx
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os

# --- CONFIGURATION ---
INPUT_SCORES = "raw_scores_dump_2.json"
INPUT_TEXT = "train/train_corpus_cleaned.txt"
OUTPUT_TRAIN_FILE = "train/train_data_taxoclass_final.json"

HIERARCHY_FILE = "class_hierarchy.txt"
CLASSES_FILE = "classes.txt"

# --- PARAMETERS ---
HARD_CONFIDENCE_FLOOR = 0.01

def load_data():
    G = nx.DiGraph()
    with open(HIERARCHY_FILE, 'r') as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) == 2: G.add_edge(int(p[0]), int(p[1]))

    id2name = {}
    with open(CLASSES_FILE, 'r') as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 2: id2name[int(p[0])] = p[1]

    roots = set([n for n, d in G.in_degree() if d == 0])

    return G, id2name, roots

def filter_conflicting_siblings(G, labels, confs):
    if not labels: return []

    # 1. Group labels by their parent(s)
    parent_to_children = defaultdict(list)
    final_labels = set(labels)

    # Map every label to its parent
    for label in labels:
        parents = list(G.predecessors(label))
        for p in parents:
            if p in final_labels:
                parent_to_children[p].append(label)

    # 2. Filter siblings
    nodes_to_remove = set()
    for parent, children in parent_to_children.items():
        if len(children) > 1:
            children.sort(key=lambda x: confs.get(x, 0.0), reverse=True)

            winner = children[0]
            losers = children[1:]
            nodes_to_remove.update(losers)

    # 3. Apply removal
    filtered_labels = [L for L in labels if L not in nodes_to_remove]
    return filtered_labels

def get_neighbors(G, node, roots):
    if node in roots:
        return roots - {node}

    # Get parent(s)
    parents = list(G.predecessors(node))

    # Collect ONLY siblings (children of my parent), NOT the parent itself
    neighbors = set()
    for p in parents:
        siblings = set(G.successors(p))
        if node in siblings:
            siblings.remove(node)
        neighbors.update(siblings)

    return neighbors

def get_descendants_of_list(G, labels):
    """Calculates Ignore Mask (Eq 7)"""
    children = set()
    for cid in labels:
        if cid in G:
            children.update(G.successors(cid))
    return list(children - set(labels))

def main():
    print("Starting TaxoClass Final Mining (With Branch Repair)...")
    G, id2name, roots = load_data()

    print(f"   Loading scores from {INPUT_SCORES}...")
    with open(INPUT_SCORES, 'r') as f:
        all_docs = json.load(f)

    # Stats Storage
    class_positive_confs = defaultdict(list)
    doc_raw_confs = {}

    # ---------------------------------------------------------
    # PASS 1: Local Confidence
    # ---------------------------------------------------------
    print("   -> Pass 1: Computing Local Confidence...")
    for doc in tqdm(all_docs):
        rid = doc['rid']
        scores = {int(k): v for k, v in doc['scores'].items()}

        current_confs = {}
        for cid, sim_score in scores.items():
            neighbors = get_neighbors(G, cid, roots)
            neighbor_scores = [scores.get(n, 0.0) for n in neighbors]
            max_neighbor = max(neighbor_scores) if neighbor_scores else 0.0

            conf = sim_score - max_neighbor
            current_confs[cid] = conf

            if conf > 0:
                class_positive_confs[cid].append(conf)

        doc_raw_confs[rid] = current_confs

    # ---------------------------------------------------------
    # PASS 2: Adaptive Medians
    # ---------------------------------------------------------
    print("   -> Pass 2: Computing Adaptive Medians...")
    class_thresholds = {}
    for cid, conf_list in class_positive_confs.items():
        if not conf_list:
            class_thresholds[cid] = 1.0
        else:
            median_val = np.median(conf_list)
            class_thresholds[cid] = max(median_val, HARD_CONFIDENCE_FLOOR)

    # ---------------------------------------------------------
    # PASS 3: Filtering & REPAIRING
    # ---------------------------------------------------------
    print("   -> Pass 3: Filtering & Repairing...")

    text_map = {}
    if os.path.exists(INPUT_TEXT):
        with open(INPUT_TEXT, 'r') as f:
            for line in f:
                p = line.strip().split('\t', 1)
                if len(p) == 2: text_map[p[0]] = p[1]

    final_dataset = []

    stats_kept = 0
    stats_repaired = 0
    stats_dropped_shallow = 0

    for doc in tqdm(all_docs):
        rid = doc['rid']
        if rid not in text_map: continue

        confs = doc_raw_confs.get(rid, {})
        core_classes = set()

        # 1. Apply Thresholds
        for cid, conf in confs.items():
            threshold = class_thresholds.get(cid, 1.0)
            if conf >= threshold:
                core_classes.add(cid)

        if not core_classes:
            continue

        # 2. Expand Ancestors (to check consistency of the whole tree)
        positive_labels = set(core_classes)
        for core in core_classes:
            try:
                positive_labels.update(nx.ancestors(G, core))
            except: pass

        # --- CHECK 1: BRANCH REPAIR ---
        subg = G.subgraph(list(positive_labels)).to_undirected()
        components = list(nx.connected_components(subg))

        if len(components) > 1:
            components.sort(key=len, reverse=True)

            best_component = components[0]
            positive_labels = set(best_component)
            stats_repaired += 1

        positive_labels = filter_conflicting_siblings(G, list(positive_labels), confs)

        # --- CHECK 2: SPECIFICITY (DEPTH FILTER) ---
        if all(c in roots for c in positive_labels):
            stats_dropped_shallow += 1
            continue

        # 3. Generate Masks
        ignore_list = get_descendants_of_list(G, positive_labels)

        final_dataset.append({
            "text": text_map[rid],
            "labels": list(positive_labels),
            "ignore": ignore_list
        })
        stats_kept += 1

    print(f"\n Complete.")
    print(f"   Original Docs: {len(all_docs)}")
    print(f"   Kept High-Quality: {stats_kept} (Repaired: {stats_repaired})")
    print(f"   Dropped (Too Shallow/Root Only): {stats_dropped_shallow}")

    # Quality Check
    if final_dataset:
        print("\n--- SAMPLE ENTRY ---")
        s = final_dataset[0]
        lbls = [id2name.get(c, c) for c in s['labels']]
        print(f"Labels: {lbls}")

    with open(OUTPUT_TRAIN_FILE, 'w') as f:
        json.dump(final_dataset, f)
    print(f"\nSaved training data to {OUTPUT_TRAIN_FILE}")

if __name__ == "__main__":
    main()