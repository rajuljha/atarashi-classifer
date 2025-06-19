# lsh_index_search_small.py

import os
import re
import random
import argparse
# import joblib
from datasketch import MinHash, MinHashLSH

# --------------------
# CONFIG
# --------------------
DATA_PATHS = ["Split-DB-Foss-Licenses", "Split-SPDX-licenses"]
NUM_PERM = 64
LSH_THRESHOLD = 0.3
# # DEFAULT_QUERY = "Redistribution and use of this software is permitted if conditions are met"
# # DEFAULT_QUERY = "Redistribution and use in source and binary forms, with or without modification, are permitted provided that the conditions below are met."
# DEFAULT_QUERY = """Attribution Assurance License Copyright (c) 2002 by AUTHOR PROFESSIONAL IDENTIFICATION * URL "PROMOTIONAL SLOGAN FOR AUTHOR'S PROFESSIONAL PRACTICE" All Rights Reserved ATTRIBUTION ASSURANCE LICENSE (adapted from the original BSD license) Redistribution and use in source and binary forms, with or without modification, are permitted provided that the conditions below are met These conditions require a modest attribution to <AUTHOR> (the "Author"), who hopes that its promotional value may help justify the thousands of dollars in otherwise billable time invested in writing this and other freely available, open-source software. 1 Redistributions of source code, in whole or part and with or without modification (the "Code"), must prominently display this GPG-signed text in verifiable form. 2 Redistributions of the Code in binary form must be accompanied by this GPG-signed text in any documentation and, each time the resulting executable program or a program dependent thereon is launched, a prominent display (e.g., splash screen or banner text) of the Author's attribution information, which includes: (a) Name ("AUTHOR"), (b) Professional identification ("PROFESSIONAL IDENTIFICATION"), and (c) URL ("URL"). 3 Neither the name nor any trademark of the Author may be used to endorse or promote products derived from this software without specific prior written permission. 4 Users are entirely responsible, to the exclusion of the Author and any other persons, for compliance with (1) regulations set by owners or administrators of employed equipment, (2) licensing terms of any other software, and (3) local regulations regarding use, including those regarding import, export, and use of encryption software. THIS FREE SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED IN NO EVENT SHALL THE AUTHOR OR ANY CONTRIBUTOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, EFFECTS OF UNAUTHORIZED OR MALICIOUS NETWORK ACCESS; PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. Attribution Assurance License Copyright (c) 2002 by AUTHOR PROFESSIONAL IDENTIFICATION * URL "PROMOTIONAL SLOGAN FOR AUTHOR'S PROFESSIONAL PRACTICE" All Rights Reserved ATTRIBUTION ASSURANCE LICENSE (adapted from the original BSD license) Redistribution and use in source and binary forms, with or without modification, are permitted provided that the conditions below are met Redistributions of source code, in whole or part and with or without modification (the "Code"), must prominently display this GPG-signed text in verifiable form. 2 Neither the name nor any trademark of the Author may be used to endorse or promote products derived from this software without specific prior written permission. 4 Attribution Assurance License Copyright (c) 2002 by AUTHOR PROFESSIONAL IDENTIFICATION * URL "PROMOTIONAL SLOGAN FOR AUTHOR'S PROFESSIONAL PRACTICE" All Rights Reserved ATTRIBUTION ASSURANCE LICENSE (adapted from the original BSD license) Redistribution and use in source and binary forms, with or without modification, are permitted provided that the conditions below are met Redistributions of the Code in binary form must be accompanied by this GPG-signed text in any documentation and, each time the resulting executable program or a program dependent thereon is launched, a prominent display (e.g., splash screen or banner text) of the Author's attribution information, which includes: (a) Name ("AUTHOR"), (b) Professional identification ("PROFESSIONAL IDENTIFICATION"), and (c) URL ("URL"). 3 Attribution Assurance License Copyright (c) 2002 by AUTHOR PROFESSIONAL IDENTIFICATION * URL "PROMOTIONAL SLOGAN FOR AUTHOR'S PROFESSIONAL PRACTICE" All Rights Reserved ATTRIBUTION ASSURANCE LICENSE (adapted from the original BSD license) Redistribution and use in source and binary forms, with or without modification, are permitted provided that the conditions below are met"""

# --------------------
# Clean text
# --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# --------------------
# Get shingles-based MinHash
# --------------------
def get_minhash(text, num_perm=NUM_PERM, shingle_size=5):
    text = text.lower().strip().replace("\n", " ")
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text)
    padded_text = " " * (shingle_size - 1) + text + " " * (shingle_size - 1)
    shingles = {padded_text[i:i+shingle_size] for i in range(len(padded_text) - shingle_size + 1)}
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf8"))
    return m

# --------------------
# Get a small, diverse file sample
# --------------------
def get_text_files(base_paths, max_licenses=10, max_files=2000):
    text_files = []

    for base_path in base_paths:
        # Always include 'AAL' license folder if it exists
        license_dirs = sorted(os.listdir(base_path))
        if 'AAL' in license_dirs:
            license_dirs.remove('AAL')
            license_dirs = ['AAL'] + license_dirs  # force AAL to be first

        selected_licenses = license_dirs[:max_licenses]
        for license_name in selected_licenses:
            license_path = os.path.join(base_path, license_name)
            if os.path.isdir(license_path):
                for file in os.listdir(license_path):
                    if file.endswith(".txt"):
                        text_files.append(os.path.join(license_path, file))

    random.seed(42)
    return random.sample(text_files, min(max_files, len(text_files)))

# --------------------
# Build the LSH index
# --------------------
def build_lsh_index(filepaths):
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    index_data = {}
    for i, path in enumerate(filepaths):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = clean_text(f.read())
                m = get_minhash(text)
                index_data[path] = m
                lsh.insert(path, m)
            if i % 200 == 0:
                print(f"[INFO] Indexed {i}/{len(filepaths)}")
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}")
    return lsh, index_data

# --------------------
# Query the index for top-k similar texts
# --------------------
def query_text(lsh, query, index_data, num_perm=NUM_PERM, top_k=5):
    m = get_minhash(clean_text(query), num_perm)
    candidates = lsh.query(m)
    scored = [(path, m.jaccard(index_data[path])) for path in candidates]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

# --------------------
# MAIN
# --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Text to query against the index")
    parser.add_argument("--max_licenses", type=int, default=10, help="Number of license types to include")
    parser.add_argument("--max_files", type=int, default=2000, help="Max number of total files")
    args = parser.parse_args()

    print(f"Sampling up to {args.max_files} files from {args.max_licenses} license types...")
    filepaths = get_text_files(DATA_PATHS, max_licenses=args.max_licenses, max_files=args.max_files)
    print(f"AAL files in index: {[f for f in filepaths if '/AAL/' in f or '\\AAL\\' in f]}")

    print(f"Building LSH index on {len(filepaths)} files...")
    lsh, index_data = build_lsh_index(filepaths)

    print(f"\nQuerying: {args.query}")
    results = query_text(lsh, args.query, index_data, top_k=5)

    if results:
        print("\nTop Matches:")
        for path, score in results:
            print(f"- {path} (score: {score:.4f})")
    else:
        print("No similar license fragments found.")
