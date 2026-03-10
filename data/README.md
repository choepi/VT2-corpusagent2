# Data Layout

- `raw/incoming/`: place original CC-News `.jsonl` or `.jsonl.gz` files here.
- `raw/ccnews_staged/`: staged subset used for a run (`debug` or `full`).
- `processed/documents.parquet`: normalized corpus.
- `indices/lexical/`: TF-IDF assets.
- `indices/dense/`: dense embedding assets.
- `indices/doc_metadata.parquet`: metadata used by retrieval and verification.

This repository keeps directories under version control but not large datasets.
