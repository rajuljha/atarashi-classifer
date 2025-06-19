# lsh_index_search.py
import os
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.pipeline import make_pipeline
import joblib
import re
import argparse


# --------------------
# CONFIG
# --------------------
DATA_PATHS = ["Split-DB-Foss-Licenses", "Split-SPDX-licenses"]
NGRAM_RANGE = (3, 5)
NUM_PERM = 128
LSH_THRESHOLD = 0.5
QUERY_TEXT = "Redistribution and use of this software is permitted if conditions are met"

# --------------------
# Helper: Collect all file paths
# --------------------
def get_text_files(base_paths):
    text_files = []
    for base_path in base_paths:
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith(".txt"):
                    text_files.append(os.path.join(root, file))
    return text_files

# --------------------
# Helper: Normalize text
# --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# --------------------
# Helper: Create MinHash signature
# --------------------
def get_minhash(text, num_perm=NUM_PERM):
    shingles = {text[i:i+5] for i in range(len(text) - 4)}
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf8"))
    return m

# --------------------
# Build LSH Index
# --------------------
def build_lsh_index(filepaths, batch_size=10000):
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    index_data = {}

    for i in range(0, len(filepaths), batch_size):
        batch = filepaths[i:i + batch_size]
        print(f"Indexing batch {i}–{i + len(batch)} of {len(filepaths)}")

        for path in batch:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = clean_text(f.read())
                    m = get_minhash(text)
                    index_data[path] = m
                    lsh.insert(path, m)
            except Exception as e:
                print(f"[ERROR] Failed on {path}: {e}")
    
    return lsh, index_data

# --------------------
# Query function
# --------------------
def query_text(lsh, text, index_data, num_perm=NUM_PERM, top_k=5):
    query_minhash = get_minhash(clean_text(text), num_perm)
    candidates = lsh.query(query_minhash)

    scored = []
    for candidate in candidates:
        score = query_minhash.jaccard(index_data[candidate])
        scored.append((candidate, score))

    # Sort by score descending and return top_k
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

# --------------------
# MAIN
# --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Load index from disk if available")
    parser.add_argument("--indexfile", default="lsh_index.pkl", help="Pickle file to save/load index")
    args = parser.parse_args()

    if args.resume and os.path.exists(args.indexfile):
        print("🔁 Loading index from disk...")
        lsh, index_data = joblib.load(args.indexfile)
    else:
        print("📥 Reading and indexing files...")
        all_files = get_text_files(DATA_PATHS)
        lsh, index_data = build_lsh_index(all_files, batch_size=10000)
        joblib.dump((lsh, index_data), args.indexfile)
        print(f"💾 Saved index to {args.indexfile}")

    print("\n🔎 Querying with test input:")
    print(f"> {QUERY_TEXT}")
    matches = query_text(lsh, QUERY_TEXT, index_data, top_k=5)
    print(f"Top {len(matches)} matches:")
    for path, score in matches:
        print(f"- {path} (score: {score:.4f})")
