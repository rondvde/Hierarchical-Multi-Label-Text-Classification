#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 22:20:05 2025

@author: Rondini Davide
"""

import torch
import json
import networkx as nx
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import re
import os
import random
import numpy as np

# --- CONFIGURATION ---
FILES = {
    "train": "train/train_corpus_cleaned.txt",
    "classes": "classes.txt",
    "hierarchy": "class_hierarchy.txt",
}
OUTPUT_DUMP = "raw_scores_dump_2.json"

# NEW MODEL
MODEL_NAME = "MoritzLaurer/roberta-base-zeroshot-v2.0-c"
SEED = 42
BATCH_SIZE_NLI = 32

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def normalize_name(name):
    return re.sub(r'[^a-z0-9]', '_', name.lower()).strip('_')

def load_data():
    id2name = {}
    with open(FILES["classes"], 'r') as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 2:
                cid = int(p[0])
                name = p[1].strip()
                id2name[cid] = name

    G = nx.DiGraph()
    with open(FILES["hierarchy"], 'r') as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) == 2: G.add_edge(int(p[0]), int(p[1]))

    roots = [n for n, d in G.in_degree() if d==0]
    return id2name, G, roots

def get_entailment_scores_batched(text, candidate_ids, id2name, model, tokenizer, device, entail_index):
    if not candidate_ids: return {}

    # Truncate text to avoid OOM or truncation errors
    text_short = text[:400]

    candidate_ids = list(set(candidate_ids))
    # Hypothesis Template matches test script
    pairs = [(text_short, f"This product is about {id2name[cid]}.") for cid in candidate_ids]
    scores_map = {}

    for i in range(0, len(pairs), BATCH_SIZE_NLI):
        batch_pairs = pairs[i:i+BATCH_SIZE_NLI]

        inputs = tokenizer(batch_pairs, padding=True, truncation=True,
                           return_tensors="pt", max_length=512).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            # Use the specific index found (0)
            batch_scores = probs[:, entail_index]

        batch_scores_np = batch_scores.float().cpu().numpy()
        for j, score in enumerate(batch_scores_np):
            cid = candidate_ids[i+j]
            scores_map[cid] = float(score)

    return scores_map

def get_candidates_taxoclass(text, roots, G, id2name, model, tokenizer, device, entail_index):
    """
    Implements Strict TaxoClass Top-Down Search (Eq 1).
    """
    # 1. INITIALIZATION
    current_level_candidates = {r: 1.0 for r in roots}
    final_scores_map = {}

    # 2. TOP-DOWN EXPANSION
    for level in range(10):
        if not current_level_candidates: break

        # A. Score current candidates
        nodes_to_score = list(current_level_candidates.keys())
        nodes_needing_nli = [n for n in nodes_to_score if n not in final_scores_map]

        if nodes_needing_nli:
            nli_scores = get_entailment_scores_batched(
                text, nodes_needing_nli, id2name, model, tokenizer, device, entail_index
            )
            final_scores_map.update(nli_scores)

        # B. Calculate Path Scores & Prune
        node_path_scores = []
        for n in nodes_to_score:
            raw_score = final_scores_map.get(n, 0.0)
            parent_path_score = current_level_candidates[n]

            # Eq 1: Path Score = Parent * Raw
            my_path_score = parent_path_score * raw_score
            node_path_scores.append((n, my_path_score))

        node_path_scores.sort(key=lambda x: x[1], reverse=True)

        # Beam Width: (Level+1)^2
        if level == 0:
            k = 3
        else:
            k = (level + 1) ** 2

        top_nodes = node_path_scores[:k]

        # C. Expand to Children
        next_level_candidates = {}

        for pid, path_score in top_nodes:
            children = list(G.successors(pid))
            if not children: continue

            if path_score < 0.001: continue

            for child in children:
                # Propagate score
                if child in next_level_candidates:
                    next_level_candidates[child] = max(next_level_candidates[child], path_score)
                else:
                    next_level_candidates[child] = path_score

        current_level_candidates = next_level_candidates

    return final_scores_map

def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"STEP 1: Hybrid Generation with {MODEL_NAME} on {device}...")

    # Load Data
    id2name, G, roots = load_data()

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # --- Find the 'entailment' value or the specific transformed i used ---
    print(f"--- CONFIG FOR: {MODEL_NAME} ---")
    print(f"ID to Label: {model.config.id2label}")

    entail_index = -1
    for idx, label in model.config.id2label.items():
        if label.lower().startswith("entail"):
            entail_index = idx
            break
        if label.lower() == "true":
            entail_index = idx
            break

    print(f"USE THIS INDEX IN CODE: entail_index = {entail_index}")
    if entail_index == -1:
        raise ValueError("could not find entailment label")
    # --------------------------------------

    if not os.path.exists(FILES["train"]):
        print(f"Error: Could not find {FILES['train']}")
        return

    with open(FILES["train"], 'r') as f:
        lines = f.readlines()

    all_data = []
    print(f"Processing {len(lines)} documents...")

    for line in tqdm(lines):
        parts = line.strip().split('\t', 1)
        if len(parts) < 2: continue
        rid, text = parts[0], parts[1]

        scores_map = get_candidates_taxoclass(
            text, roots, G, id2name, model, tokenizer, device, entail_index
        )

        valid_scores = {c: s for c, s in scores_map.items() if s > 0.001}

        if valid_scores:
            all_data.append({"rid": rid, "scores": valid_scores})

    print(f"Saving to {OUTPUT_DUMP}...")
    with open(OUTPUT_DUMP, 'w') as f:
        json.dump(all_data, f)

if __name__ == "__main__":
    main()
