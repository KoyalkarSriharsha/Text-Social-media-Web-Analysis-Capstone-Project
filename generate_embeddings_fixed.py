#!/usr/bin/env python3
"""
generate_embeddings_fixed.py

Compute and save SBERT embeddings for BERTopic + Chatbot retrieval.

Usage:
    python generate_embeddings_fixed.py --input ./data/processed/processed_records.csv --outdir ./data/embeddings --model all-MiniLM-L6-v2 --sample 3000
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def compute_embeddings(texts, model_name='all-MiniLM-L6-v2', batch_size=64):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("Install: pip install sentence-transformers") from e

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True
    )
    return embeddings

def main(input_csv, outdir, model_name='all-MiniLM-L6-v2', sample_n=None, force=False):
    input_csv = Path(input_csv)
    outdir = Path(outdir)
    ensure_dir(outdir)

    if not input_csv.exists():
        print("Input CSV not found:", input_csv)
        return

    df = pd.read_csv(input_csv)
    if sample_n:
        df = df.head(sample_n).reset_index(drop=True)

    texts = (df['title'].fillna('') + '. ' + df['content'].fillna('')).astype(str).tolist()
    ids = df['record_id'].astype(str).tolist()
    titles = df['title'].fillna('').astype(str).tolist()

    emb_all = outdir / 'embeddings_all-mini.npy'
    emb_chat = outdir / 'chatbot_embeddings.npy'
    meta_path = Path('./data/results') / 'retrieval_index_meta.json'

    if emb_all.exists() and emb_chat.exists() and meta_path.exists() and not force:
        print("Embeddings already exist. Use --force to regenerate.")
        return

    print(f"Computing embeddings using {model_name} for {len(texts)} documents...")
    embeddings = compute_embeddings(texts, model_name=model_name)
    print("Saving embeddings...")

    np.save(str(emb_all), embeddings)
    np.save(str(emb_chat), embeddings)

    ensure_dir(meta_path.parent)
    meta = {'record_ids': ids, 'titles': titles}
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print("Done! Embeddings saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./data/processed/processed_records.csv')
    parser.add_argument('--outdir', type=str, default='./data/embeddings')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    main(
        args.input,
        args.outdir,
        model_name=args.model,
        sample_n=args.sample,
        force=args.force
    )
