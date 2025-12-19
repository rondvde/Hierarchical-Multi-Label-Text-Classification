#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 23:23:51 2025

@author: Rondini Davide
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import numpy as np
import json
import networkx as nx
import scipy.sparse as sp
from tqdm import tqdm
import os

# --- CONFIGURATION ---
ITERATIONS = 3
EPOCHS_SUPERVISED = 3
EPOCHS_PER_ITER = 1
BATCH_SIZE = 32
LR_ENCODER = 1e-6
LR_CLASSIFIER = 5e-4 
FILES = {
    "train_text": "train/train_corpus_cleaned.txt",
    "silver_data": "train/train_data_combined.json",
    "hierarchy": "class_hierarchy.txt",
    "class_emb": "class_embeddings.pt"
}
MODEL_NAME = 'bert-base-uncased'
NUM_CLASSES = 531
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. DATASETS
# ==========================================

class LabeledDataset(Dataset):
    """PHASE 1: Core Class Guided Training (Eq. 7 & 8)"""
    def __init__(self, json_file, tokenizer, max_len, num_classes):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label_vec = torch.zeros(self.num_classes)
        label_vec[item['labels']] = 1.0
        
        # Loss Mask: 0 for "Ignored" classes (Children), 1 otherwise
        mask_vec = torch.ones(self.num_classes)
        if 'ignore' in item:
            mask_vec[item['ignore']] = 0.0
        mask_vec[item['labels']] = 1.0

        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_vec,
            'loss_mask': mask_vec
        }

class UnlabeledDataset(Dataset):
    """PHASE 2: Self-Training (Eq. 9 & 10)"""
    def __init__(self, text_list, tokenizer, max_len, num_classes):
        self.texts = text_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Init soft targets Q with zeros
        self.soft_targets = torch.zeros(len(text_list), num_classes)

    def __len__(self):
        return len(self.texts)

    def update_targets(self, new_targets):
        """Updates the Q distribution targets"""
        self.soft_targets = new_targets

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'soft_targets': self.soft_targets[idx]
        }

# ==========================================
# 2. MODEL
# ==========================================

def build_adjacency(hierarchy_file, num_classes, device):
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
    return torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj.shape)).to(device)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias

class TaxoClassModel(nn.Module):
    def __init__(self, model_name, num_classes, hierarchy_file, class_emb_file):
        super(TaxoClassModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.gc1 = GraphConvolution(768, 768)
        self.gc2 = GraphConvolution(768, 768)
        self.bilinear = nn.Linear(768, 768, bias=False)
        self.adj = build_adjacency(hierarchy_file, num_classes, DEVICE)
        self.class_features = nn.Parameter(torch.load(class_emb_file).float(), requires_grad=True)

    def forward(self, input_ids, attention_mask):
        doc_out = self.bert(input_ids, attention_mask=attention_mask)
        doc_emb = self.dropout(doc_out.pooler_output)
        x = torch.relu(self.gc1(self.class_features, self.adj))
        class_emb = self.gc2(x, self.adj)
        doc_trans = self.bilinear(doc_emb)
        logits = torch.matmul(doc_trans, class_emb.t())
        return logits

# ==========================================
# 3. UTILS (Eq 10 Calculation)
# ==========================================

def calculate_target_distribution(probs):
    """
    Implements Equation 10.
    Enhances high-confidence predictions while down-weighting low-confidence ones.
    """
    # 1. f_j: Class frequencies (sum over batch/corpus)
    f_j = probs.sum(axis=0, keepdims=True).clamp(min=1e-9)

    # 2. Term Positive: p^2 / f_j
    term_pos = probs.pow(2) / f_j

    # 3. Term Negative: (1-p)^2 / f_neg
    probs_neg = 1 - probs
    f_neg = probs_neg.sum(axis=0, keepdims=True).clamp(min=1e-9)
    term_neg = probs_neg.pow(2) / f_neg

    # 4. Q = TermPos / (TermPos + TermNeg)
    q_final = term_pos / (term_pos + term_neg + 1e-9)
    return q_final

# ==========================================
# 4. TRAINING FUNCTIONS
# ==========================================

def train_supervised_phase(model, train_loader, optimizer, device):
    """PHASE 1: Core Class Guided Training"""
    model.train()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    loop = tqdm(train_loader, desc="[Supervised]")
    for batch in loop:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss_mask = batch['loss_mask'].to(device)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss_matrix = criterion(logits, labels)
        
        # Apply mask to ignore children classes
        masked_loss = loss_matrix * loss_mask
        loss = masked_loss.sum() / loss_mask.sum().clamp(min=1)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)

def train_self_training_phase(model, dataset, batch_size, optimizer, device):
    """
    PHASE 2: Self-Training. 
    1. Generates predictions P.
    2. Calculates targets Q (Eq 10).
    3. Trains on KL(Q||P) via BCE (Eq 9).
    """
    model.train()

    # --- Step A: Update Q (Target Distribution) ---
    # The paper updates Q periodically (e.g. every 25 batches).
    # For efficiency on the full dataset, we update Q at the start of the iteration.
    
    print("   -> Updating Q (Target Distribution)...")
    eval_loader = DataLoader(dataset, batch_size=batch_size*2, shuffle=False)
    all_probs = []
    
    model.eval() 
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Calculating Q"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(ids, mask)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
    
    full_probs = torch.cat(all_probs, dim=0)
    new_targets_Q = calculate_target_distribution(full_probs) # Eq 10 
    dataset.update_targets(new_targets_Q)
    model.train()

    # --- Step B: Train Classifier on Q ---
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Using BCEWithLogitsLoss with soft targets is equivalent to minimizing 
    criterion = nn.BCEWithLogitsLoss() 

    total_loss = 0
    loop = tqdm(train_loader, desc="[Self-Train]")

    for batch in loop:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        targets_Q = batch['soft_targets'].to(device)

        optimizer.zero_grad()
        logits = model(ids, mask)
        
        # Train to match the sharpened distribution Q
        loss = criterion(logits, targets_Q)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main():
    print(f" Starting TaxoClass Training Pipeline...")
    torch.manual_seed(42)

    # 1. Load Texts
    print("   Loading text corpus...")
    all_texts = []
    if os.path.exists(FILES["train_text"]):
        with open(FILES["train_text"], 'r') as f:
            for line in f:
                p = line.strip().split('\t', 1)
                if len(p) == 2: all_texts.append(p[1].strip())
    else:
        print(f"Error: {FILES['train_text']} not found.")
        return

    # 2. Initialize Model
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = TaxoClassModel(MODEL_NAME, NUM_CLASSES, FILES["hierarchy"], FILES["class_emb"])
    model.to(DEVICE)

    # Optimizer with differential Learning Rates 
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': LR_ENCODER},
        {'params': model.gc1.parameters(), 'lr': LR_CLASSIFIER},
        {'params': model.gc2.parameters(), 'lr': LR_CLASSIFIER},
        {'params': model.bilinear.parameters(), 'lr': LR_CLASSIFIER},
        {'params': [model.class_features], 'lr': LR_CLASSIFIER}
    ])

    # ---------------------------------------------------------
    # PART 1: Supervised Training (Silver Labels)
    # ---------------------------------------------------------
    if os.path.exists(FILES["silver_data"]):
        print(f"\nPhase 1: Supervised Training (Eq. 8) for {EPOCHS_SUPERVISED} epochs...")
        labeled_dataset = LabeledDataset(FILES["silver_data"], tokenizer, 128, NUM_CLASSES)
        labeled_loader = DataLoader(labeled_dataset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(EPOCHS_SUPERVISED):
            loss = train_supervised_phase(model, labeled_loader, optimizer, DEVICE)
            print(f"   Epoch {epoch+1}: Loss = {loss:.4f}")

        if not os.path.exists("saved_models"): os.makedirs("saved_models")
        torch.save(model.state_dict(), "saved_models/taxoclass_pretrained.pt")
    else:
        print("CRITICAL ERROR: Silver data not found. Cannot start.")
        return

    # ---------------------------------------------------------
    # PART 2: Self-Training (Eq 9 & 10)
    # ---------------------------------------------------------
    print(f"\nPhase 2: Multi-Label Self-Training for {ITERATIONS} iterations...")
    
    # Create dataset with ALL texts (labeled + unlabeled mixed)
    unlabeled_dataset = UnlabeledDataset(all_texts, tokenizer, 128, NUM_CLASSES)

    for iteration in range(ITERATIONS):
        print(f"\n--- Self-Training Iteration {iteration + 1} / {ITERATIONS} ---")

        # In each iteration, we update Q and train for a few epochs
        for ep in range(EPOCHS_PER_ITER):
            loss = train_self_training_phase(model, unlabeled_dataset, BATCH_SIZE, optimizer, DEVICE)
            print(f"   Self-Train Epoch {ep+1}: Loss = {loss:.4f}")

    # Save Final Model
    torch.save(model.state_dict(), "saved_models/taxoclass_final_sl_llm.pt")
    print("\nTraining Complete. Model saved to 'saved_models/taxoclass_final_sl_llm.pt'")

if __name__ == "__main__":
    main()
