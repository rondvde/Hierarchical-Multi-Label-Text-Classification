import json
import random
import os
import hashlib

# --- CONFIGURATION ---
SILVER_FILE = "train/train_data_taxoclass_final.json"
GOLD_FILE = "annotated_data.json"
OUTPUT_FILE = "train/train_data_combined.json"
GOLD_WEIGHT_FACTOR = 5 

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_text_hash(text):
    clean_text = "".join(text.split()).lower()
    return hashlib.md5(clean_text.encode('utf-8')).hexdigest()

def main():
    print(f"Loading data...")
    silver_data = load_json(SILVER_FILE)
    gold_data = load_json(GOLD_FILE)
    
    print(f"   Silver raw count: {len(silver_data)}")
    print(f"   Gold raw count:   {len(gold_data)}")

    # 1. Process Gold Data
    gold_text_hashes = set()
    cleaned_gold = []
    
    for item in gold_data:
        text = item.get('text', item.get('doc_content', ''))
        
        if text:
            gold_text_hashes.add(get_text_hash(text))
        
        cleaned_gold.append({
            "id": str(item.get('id', f"gold_{random.randint(0,1000000)}")),
            "text": text,
            "labels": item['labels'] 
        })

    # 2. Filter Silver Data
    unique_silver = []
    skipped_count = 0
    
    for i, item in enumerate(silver_data):
        text = item.get('text', item.get('doc_content', ''))
        
        # Check if this text is already in our Gold set
        if get_text_hash(text) in gold_text_hashes:
            skipped_count += 1
            continue
            
        unique_silver.append({
            "id": str(item.get('id', item.get('doc_id', f"silver_{i}"))),
            "text": text,
            "labels": item['labels']
        })
            
    print(f"   Removed {skipped_count} silver samples that overlapped with Gold.")
    print(f"   Silver (Filtered): {len(unique_silver)}")

    # 3. Oversample Gold Data (x5)
    weighted_gold = cleaned_gold * GOLD_WEIGHT_FACTOR
    print(f"   Gold (Oversampled x{GOLD_WEIGHT_FACTOR}): {len(weighted_gold)}")

    # 4. Combine
    combined_data = unique_silver + weighted_gold
    
    # 5. Shuffle
    random.shuffle(combined_data)

    # 6. Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2)

    print(f"Successfully created {OUTPUT_FILE} with {len(combined_data)} samples.")

if __name__ == "__main__":
    main()