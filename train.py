#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 18:54:32 2025

@author: daviderondini
"""

import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from dataset import ReviewDataset 
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import f1_score

# --- 1. CONFIGURATION ---
SEED = 42
BATCH_SIZE = 16    
EPOCHS = 4        
LEARNING_RATE = 2e-5
MAX_LEN = 128      
NUM_CLASSES = 531  
MODEL_NAME = 'bert-base-uncased'
OUTPUT_DIR = "saved_models"

FILES = {
    "train": "train/train_data.json",
    "val":   "test/val_data.json" 
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(SEED)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 2. DEVICE SETUP ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

# --- 3. DATA LOADING ---
print("Loading Tokenizer and Dataset...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

try:
    print(f"Loading Train: {FILES['train']}")
    train_dataset = ReviewDataset(FILES["train"], tokenizer, max_len=MAX_LEN, num_classes=NUM_CLASSES)
    
    print(f"Loading Val:   {FILES['val']}")
    val_dataset = ReviewDataset(FILES["val"], tokenizer, max_len=MAX_LEN, num_classes=NUM_CLASSES)
except FileNotFoundError as e:
    print(f"\nFile/s not found.\n{e}")
    exit()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples:   {len(val_dataset)}")

# --- 4. MODEL INITIALIZATION  ---
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    problem_type="multi_label_classification"
)
model.to(device)

# --- 5. OPTIMIZER AND SCHEDULER ---
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0, 
    num_training_steps=total_steps
)

# metrics
def compute_metrics(preds, labels):
    probs = torch.sigmoid(preds)
    
    if random.random() < 0.05:
        print(f"\n[DEBUG] Max Prob: {probs.max().item():.4f} | Mean Prob: {probs.mean().item():.4f}")
    
    threshold = 0.3 
    preds_binary = (probs > threshold).float().cpu().numpy()
    labels_numpy = labels.cpu().numpy()
    
    f1_micro = f1_score(labels_numpy, preds_binary, average='micro', zero_division=0)
    f1_macro = f1_score(labels_numpy, preds_binary, average='macro', zero_division=0)
    return f1_micro, f1_macro

# --- 6. TRAINING LOOP ---
best_val_f1 = 0.0

print("\n--- TRAINING ---")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        model.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    
    # --- VALIDATION LOOP ---
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    print("Validation...")
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            val_loss += outputs.loss.item()
            all_preds.append(outputs.logits)
            all_labels.append(labels)
            
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    val_f1_micro, val_f1_macro = compute_metrics(all_preds, all_labels)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"\nStats Epoch {epoch+1}:")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss:   {avg_val_loss:.4f}")
    print(f"Val F1 (Micro): {val_f1_micro:.4f}")
    print(f"Val F1 (Macro): {val_f1_macro:.4f}")
    
    # Saving best model
    if val_f1_micro > best_val_f1:
        best_val_f1 = val_f1_micro
        save_path = os.path.join(OUTPUT_DIR, "best_model_silver.pt")
        torch.save(model.state_dict(), save_path)
        print(f"New best model saved in {save_path}")

print("\nTRAINING COMPLETED.")