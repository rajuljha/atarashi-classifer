# Atarashi Classifier - License Text Similarity Detection

A Python-based project that implements Locality Sensitive Hashing (LSH) with SimHash algorithm for efficient license text similarity detection. This project is designed to help identify similar software licenses by comparing their text content using advanced embedding and hashing techniques.

## TLDR
### Points to note:
- Identified all the licenses that were indexed correctly. (46 licenses in 10000 file samples)
- There were (46/654) licenses that were indexed into LSH.
- Identified all the non license text correclty. (20/674) samples.
- 608 licenses were not indexed, that is they were not part of the search space. Even then, it identified some statements that it had not seen previously. (203/608)
- Overall accuracy even with only (10000/162833) files indexed.

## Project Overview

This project implements a license text similarity detection system using:
- Sentence transformers for text embedding
- SimHash algorithm for dimensionality reduction
- Locality Sensitive Hashing (LSH) for efficient similarity search
- Combined-Licenses dataset from SPDX and FOSS license collections

## Dataset Information

The project uses a Combined-Licenses dataset that merges two main sources merged from the [Minerva Dataset](https://github.com/fossology/Minerva-Dataset-Generation/):
- Split-DB-Foss-Licenses
- Split-SPDX-Licenses

The dataset contains various license texts organized by license type. Some key statistics:
- Total number of unique licenses: 90+
- Most common licenses include APL-1.0, BitTorrent-1.1, RPL-1.1, etc.
- Each license category contains multiple text variations
- Files are stored in .txt format

## Project Structure

```
atarashi_classifier/
├── LSH.py                 # LSH and SimHash implementation
├── scripts/
│   ├── Dataset.ipynb     # Dataset processing and analysis
│   └── Test_SimHash.ipynb # Testing and evaluation scripts
├── Combined-Licenses/     # Combined dataset directory
├── plots/                 # Visualization plots
└── cache/                # Cached embeddings and LSH indexes
```

## Setup Instructions

> [!IMPORTANT]
Due to massive size of the dataset (~7GBs, cloning the repo might take some time)

1. Clone the repository:
```bash
git clone https://github.com/rajuljha/atarashi_classifier.git
cd atarashi_classifier
```

2. Set up the environment using `uv`:
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows

# Install dependencies
uv pip install -r requirements.txt
```

3. Required Dependencies(see the pyproject.toml file for more details):
- sentence-transformers
- numpy
- pandas
- joblib
- tqdm
- pathlib

## Implementation Details

### LSH with SimHash
The project implements Locality Sensitive Hashing using SimHash algorithm (see `LSH.py`):
- SimHash reduces high-dimensional vectors to binary signatures
- Multiple hash tables are used to increase the probability of finding similar items
- The implementation uses random projections for hash function generation

### Text Processing Pipeline
1. License texts are embedded using the "all-MiniLM-L6-v2" sentence transformer
2. Dense vectors are generated for each license text
3. LSH index is built using these vectors
4. Similarity queries use the same embedding process for comparison

## Results

The project includes evaluation results in `license_detection_results.csv`, which contains:
- Similarity scores between license texts
- Detection accuracy metrics
- Performance benchmarks

Visualization plots in the `plots/` directory show:
- Distribution of license types
- Similarity score distributions
- Performance metrics

## Usage

To run similarity detection on license texts:

```python
from LSH import LSH
from sentence_transformers import SentenceTransformer

# Initialize models
model = SentenceTransformer("all-MiniLM-L6-v2")
lsh = LSH(hash_size=32, input_dim=384, num_tables=30)

# Add license texts to the index
vector = model.encode(license_text)
lsh.add(vector, license_name)

# Query similar licenses
query_vector = model.encode(query_text)
similar_licenses = lsh.query(query_vector)
```

## Performance

The system demonstrates efficient similarity detection:
- Fast query times due to LSH-based indexing
- Scalable to large license collections
- Configurable precision-recall tradeoff through LSH parameters


## License

This project is licensed under [General Public License 2.0](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
