#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 15:04:04 2025

@author: Rondini Davide
"""

import re
from langdetect import detect, LangDetectException
from tqdm import tqdm
import random
import numpy as np
import torch

# --- CONFIGURATION ---
INPUT_FILE = "train/train_corpus.txt"
OUTPUT_FILE = "train/train_corpus_cleaned.txt"

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def is_gibberish(text):
    words = text.split()
    
    if not words:
        return True
        
    # keyboard smashing detector (if word is longer than 30 characters)
    for word in words:
        if len(word) > 30: 
            return True
            
    # if word has few unique character compared to its length (es. "hahahahaha")
    if len(set(text)) < 3 and len(text) > 10:
        return True

    return False

def is_english(text):
    try:
        if detect(text) == 'en':
            return True
    except LangDetectException:
        return False
    return False

def clean_corpus():
    set_seed(42)
    print("--- STARTING DATA CLEANING ---")
    
    clean_lines = []
    removed_count = 0
    gibberish_count = 0
    non_english_count = 0
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE} not found.")
        return

    print(f"Total reviews: {len(lines)}")
    
    for line in tqdm(lines, desc="Processing reviews"):
        parts = line.strip().split('\t', 1)
        
        # format control (ID + Text)
        if len(parts) < 2:
            removed_count += 1
            continue
            
        rid, text = parts[0], parts[1]
        
        # 1. Basic initial cleaning
        # removing punctuation for control, but saving in final dataset
        text_check = re.sub(r'[^\w\s]', '', text) 
        
        # 2. control minimum length
        if len(text_check.split()) < 3:
            removed_count += 1
            continue

        # 3. gibberish control
        if is_gibberish(text_check):
            gibberish_count += 1
            continue
            
        # 4. english language control
        if not is_english(text):
            non_english_count += 1
            continue
            #we keep if all tests are passed
        clean_lines.append(line)

    # saving
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.writelines(clean_lines)

    print("\n--- CLEANING RESULTS ---")
    print(f"File saved in: {OUTPUT_FILE}")
    print(f"reviews kept: {len(clean_lines)}")
    print(f"reviews removed (total): {len(lines) - len(clean_lines)}")
    print(f"  - Not in english: {non_english_count}")
    print(f"  - Gibberish/Nonsense: {gibberish_count}")
    print(f"  - Wrong fromat/Too short: {removed_count}")

if __name__ == "__main__":
    clean_corpus()
