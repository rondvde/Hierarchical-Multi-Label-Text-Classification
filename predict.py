#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 23:29:51 2025

@author: Rondini Davide
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import networkx as nx
import scipy.sparse as sp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import csv

# --- CONFIGURATION ---
MODEL_PATH = "saved_models/taxoclass_final_sl_llm.pt"
TEST_FILE = "test/test_corpus.txt"
OUTPUT_CSV = "2025952015_final.csv"
HIERARCHY_FILE = "class_hierarchy.txt"
CLASS_EMB = "class_embeddings.pt"
MODEL_NAME = 'bert-base-uncased'
NUM_CLASSES = 531
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_adjacency(hierarchy_file, num_classes):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_classes))
    with open(hierarchy_file, 'r') as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) == 2:
                u, v = int(p[0]), int(p[1])
                G.add_edge(u, v)
                G.add_edge(v, u)

    adj = nx.adjacency_matrix(G) + sp.eye(num_classes)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

    norm_adj = norm_adj.tocoo()
    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data).float()
    return torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj.shape)).to(DEVICE)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias

class TaxoClassModel(nn.Module):
    def __init__(self, model_name, num_classes, hierarchy_file, class_emb_file):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.gc1 = GraphConvolution(768, 768)
        self.gc2 = GraphConvolution(768, 768)
        self.adj = build_adjacency(hierarchy_file, num_classes)
        self.class_features = nn.Parameter(torch.zeros(num_classes, 768))
        self.bilinear = nn.Linear(768, 768, bias=False)

    def forward(self, input_ids, attention_mask):
        doc_out = self.bert(input_ids, attention_mask=attention_mask)
        doc_emb = self.dropout(doc_out.pooler_output)
        x = F.relu(self.gc1(self.class_features, self.adj))
        class_emb = self.gc2(x, self.adj)
        doc_trans = self.bilinear(doc_emb)
        logits = torch.matmul(doc_trans, class_emb.t())
        return logits

class InferenceDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines and not lines[0].split('\t')[0].isdigit():
                lines = lines[1:]
            for line in lines:
                p = line.strip().split('\t', 1)
                if len(p) == 2: self.data.append((p[0], p[1]))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        rid, text = self.data[idx]
        enc = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        return rid, enc['input_ids'].flatten(), enc['attention_mask'].flatten()

def get_predictions_smart(probs, threshold=0.15):
    top_indices = np.argsort(probs)[::-1]
    candidates = [i for i in top_indices if probs[i] > threshold]

    # If nothing > 0.15, take Top 2 regardless of score
    if len(candidates) < 2:
        candidates = top_indices[:2].tolist()

    # 3. Cap at Top 3
    if len(candidates) > 3:
        candidates = candidates[:3]

    return sorted(candidates)

def main():
    print(f"Loading trained model from {MODEL_PATH}...")

    if not os.path.exists(MODEL_PATH):
        print("Model file not found")
        return

    model = TaxoClassModel(MODEL_NAME, NUM_CLASSES, HIERARCHY_FILE, CLASS_EMB)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    ds = InferenceDataset(TEST_FILE, tokenizer, MAX_LEN)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    results = []
    print("   Running Inference...")

    with torch.no_grad():
        for rids, ids, mask in tqdm(loader):
            ids, mask = ids.to(DEVICE), mask.to(DEVICE)
            logits = model(ids, mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            for i, rid in enumerate(rids):
                preds = get_predictions_smart(probs[i])
                pred_str = ",".join(map(str, preds))
                results.append([rid, pred_str])

    results.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows(results)

    print(f"Saved submission to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()