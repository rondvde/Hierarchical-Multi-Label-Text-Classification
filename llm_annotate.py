#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 18:30:12 2025

@author: Rondini Davide
"""

import google.generativeai as genai
import json
import time
import os
import networkx as nx
import random
from tqdm import tqdm
from google.api_core.exceptions import ResourceExhausted, InternalServerError

# --- CONFIGURATION ---
API_KEY = "APIKEY" # I REMOVED MY APIKEY FOR SUBMISSION
MODEL_NAME = "gemini-2.5-flash-lite"

FILES = {
    "input_corpus": "train/train_corpus_cleaned.txt",
    "classes": "classes.txt",
    "hierarchy": "class_hierarchy.txt"
}
OUTPUT_FILE = "annotated_data.json"
LOG_FILE = "llm_history.log" 

SAMPLE_SIZE = 1000
RANDOM_SEED = 42

# --- NORMALIZATION OF TEXT ---
def normalize_text(text):
    return text.lower().strip().replace("_", " ")

# --- 1. SETUP RESOURCES ---
def load_resources():
    G = nx.DiGraph()
    try:
        with open(FILES["hierarchy"], 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    G.add_edge(int(parts[0]), int(parts[1]))
    except FileNotFoundError:
        print(f"Error: {FILES['hierarchy']} not found.")
        exit()

    label_to_id = {} 
    all_labels_list = []
    
    try:
        with open(FILES["classes"], 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    cid = int(parts[0])
                    raw_name = parts[1]
                    clean_key = normalize_text(raw_name)
                    label_to_id[clean_key] = cid
                    all_labels_list.append(raw_name.strip())
    except FileNotFoundError:
        print(f"Error: {FILES['classes']} not found.")
        exit()
        
    return G, label_to_id, all_labels_list

def propagate_labels(leaf_ids, G):
    final_set = set(leaf_ids)
    for label in list(final_set):
        if label in G:
            ancestors = nx.ancestors(G, label)
            final_set.update(ancestors)
    return list(final_set)

# --- 2. SETUP GEMINI ---
def get_model(all_labels_list):
    genai.configure(api_key=API_KEY)
    categories_str = ", ".join(all_labels_list)
    
    sys_instruction = f"""
    You are an expert annotator.
    ALLOWED CATEGORIES: [{categories_str}]
    TASK: Classify the review strictly using ONLY the category names provided above.
    OUTPUT FORMAT RULES:
    1. Name ONE ONLY most related category
    2. Example output: baby food
    3. Do NOT use brackets, quotes, or JSON. Just text.
    """
    
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=sys_instruction,
        generation_config={"temperature": 0.0, "max_output_tokens": 50}
    )

# --- 3. DATA LOADING & SAMPLING ---
def load_and_sample_data():
    print(f"Loading raw corpus from {FILES['input_corpus']}...")
    all_reviews = []
    
    try:
        with open(FILES['input_corpus'], 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    all_reviews.append({'id': parts[0], 'text': parts[1]})
    except FileNotFoundError:
        print(f"Input corpus not found at {FILES['input_corpus']}")
        exit()

    print(f"Total reviews available: {len(all_reviews)}")
    
    random.seed(RANDOM_SEED)
    selected_reviews = random.sample(all_reviews, SAMPLE_SIZE)
        
    return selected_reviews

# --- 4. MAIN LOOP ---
def main():
    print("Loading resources...")
    hierarchy_graph, label_to_id, all_labels_list = load_resources()
    model = get_model(all_labels_list)
    
    reviews_to_process = load_and_sample_data() 

    annotated_reviews = []
    existing_ids = set()
    
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                annotated_reviews = json.load(f)
                existing_ids = {item['id'] for item in annotated_reviews}
            print(f"Resumed: {len(annotated_reviews)} already done in output file.")
        except:
            pass

    final_queue = [r for r in reviews_to_process if r['id'] not in existing_ids]
    print(f"Starting annotation on {len(final_queue)} remaining reviews...")

    log_file = open(LOG_FILE, "a", encoding="utf-8")
    save_interval = 10
    counter = 0

    for review in tqdm(final_queue):
        text = review['text']
        r_id = review['id']
        
        prompt = f"""Review: "{text}" """
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                time.sleep(4) 
                
                response = model.generate_content(prompt)
                raw_text = response.text.strip()
                
                if raw_text.endswith('.'): raw_text = raw_text[:-1]
                clean_name = normalize_text(raw_text)

                found_ids = []
                if clean_name in label_to_id:
                    found_ids.append(label_to_id[clean_name])
                else:
                     log_file.write(f"[WARN] ID {r_id}: Unknown class '{raw_text}'\n")

                full_labels = propagate_labels(found_ids, hierarchy_graph)
                
                annotated_reviews.append({
                    "id": r_id,
                    "text": text,
                    "labels": full_labels,
                    "origin": "gemini_random_sample"
                })
                break 

            except Exception as e:
                if attempt == max_retries - 1:
                    log_file.write(f"[ERROR] ID {r_id}: {e}\n")
                    annotated_reviews.append({"id": r_id, "text": text, "labels": [], "error": str(e)})
                time.sleep(10)
        
        counter += 1
        if counter % save_interval == 0:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(annotated_reviews, f, indent=4)
            log_file.flush()

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(annotated_reviews, f, indent=4)
    
    log_file.close()
    print(f"Finished. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()