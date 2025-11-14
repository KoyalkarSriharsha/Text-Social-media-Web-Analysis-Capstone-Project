#!/usr/bin/env python3
"""
preprocess_and_ingest_to_csv.py

Preprocessing pipeline script for TruthLens synthetic dataset that writes CSV outputs.

What it does (high level):
1. Loads synthetic JSONL dataset produced by synthetic_truthlens_generator.py
   (default path: ./data/raw/synthetic_all_sources_v1.jsonl)
2. Basic cleaning and normalization (HTML strip, whitespace)
3. Language detection (langdetect)
4. Tokenization, sentence segmentation, optional NER using spaCy (if installed)
5. Simple claim extraction heuristics (regex + sentence-level)
6. Writes CSV outputs in ./data/processed/:
   - processed_records.csv  (one row per source record)
   - raw_records.csv        (flattened small raw view for auditing)
   - entities.csv           (one row per extracted entity)
   - claim_candidates.csv   (one row per claim candidate)
   - _metadata.json         (dataset metadata)

Usage:
    python preprocess_and_ingest_to_csv.py --input ./data/raw/synthetic_all_sources_v1.jsonl

Options:
    --input PATH        Path to input JSONL (default: ./data/raw/synthetic_all_sources_v1.jsonl)
    --outdir PATH       Output directory for CSVs (default: ./data/processed)
    --chunk INT         Chunk size for writing CSVs (default: 1000)
    --skip-ner          Force skip spaCy NER even if spaCy is installed
    --no-raw-csv        Skip writing raw_records.csv (optional)
    --gzip              Write compressed CSVs (.gz) to save space (default: False)
    --sample N          Process only first N records (for quick tests)

Notes:
    - This script is intended for development/demo. For production, use batching, streaming,
      and careful error handling.
    - If spaCy is installed and you want higher-quality NER, install it (Conda recommended):
        conda install -c conda-forge spacy
        python -m spacy download en_core_web_sm
"""

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from langdetect import detect, DetectorFactory
from tqdm import tqdm
import pandas as pd

# Optional imports
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    spacy = None
    SPACY_AVAILABLE = False

DetectorFactory.seed = 0  # deterministic language detection

# -------------------------
# Utility functions
# -------------------------
def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Input file not found: {path}')
    docs = []
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except Exception as e:
                # skip malformed lines but print a message
                print('Warning: failed to parse a JSON line:', e)
    return docs

def clean_text(text):
    if text is None:
        return ''
    # remove html tags (basic)
    text = re.sub(r'<[^>]+>', ' ', text)
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def safe_detect_lang(text):
    try:
        return detect(text)
    except Exception:
        return 'unknown'

def extract_claims_simple(text):
    """
    Very simple heuristic to extract candidate claims:
      - sentences containing numeric patterns, percentages, dates, or words like 'study', 'found', 'claims'
    Returns list of claim texts (strings).
    """
    claims = []
    if not text:
        return claims
    # split to sentences using simple split (spaCy will do better if available)
    sents = re.split(r'(?<=[.!?])\s+', text)
    for s in sents:
        if re.search(r'\b(study|research|found|claims|claim|reported|reports|percent|%|\d{4})\b', s, flags=re.I):
            s_clean = s.strip()
            if len(s_clean) > 20:
                claims.append(s_clean)
    return claims

# -------------------------
# Main processing function
# -------------------------
def preprocess_to_csv(input_path,
                      out_dir='./data/processed',
                      chunk_size=1000,
                      skip_ner=False,
                      write_raw_csv=True,
                      gzip=False,
                      sample_n=None):
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_csv = out_dir / ('processed_records.csv.gz' if gzip else 'processed_records.csv')
    raw_csv = out_dir / ('raw_records.csv.gz' if gzip else 'raw_records.csv')
    entities_csv = out_dir / ('entities.csv.gz' if gzip else 'entities.csv')
    claims_csv = out_dir / ('claim_candidates.csv.gz' if gzip else 'claim_candidates.csv')
    metadata_json = out_dir / '_metadata.json'

    print('Loading JSONL from:', input_path)
    docs = load_jsonl(input_path)
    if sample_n:
        docs = docs[:sample_n]
    n_docs = len(docs)
    print(f'Loaded {n_docs} documents.')

    # spaCy setup (optional)
    nlp = None
    use_ner = False
    if not skip_ner and SPACY_AVAILABLE:
        try:
            nlp = spacy.load('en_core_web_sm')
            use_ner = True
            print('spaCy loaded for NER/tokenization.')
        except Exception as e:
            print('spaCy present but failed to load model en_core_web_sm. NER disabled. Error:', e)
            nlp = None
            use_ner = False
    else:
        if skip_ner:
            print('Skipping spaCy NER due to --skip-ner flag.')
        else:
            print('spaCy not installed; NER/tokenization disabled.')

    # Prepare chunked writing
    first_proc_chunk = True
    first_raw_chunk = True
    first_entities_chunk = True
    first_claims_chunk = True

    proc_buffer = []
    raw_buffer = []
    entities_buffer = []
    claims_buffer = []

    for i, raw_doc in enumerate(tqdm(docs, desc='Processing'), start=1):
        # Basic raw entry (for optional raw CSV)
        raw_entry = {
            'record_id': raw_doc.get('record_id'),
            'source_type': raw_doc.get('source_type'),
            'source_name': raw_doc.get('source_name'),
            'published_at': raw_doc.get('published_at'),
            'collected_at': raw_doc.get('collected_at'),
            'title': raw_doc.get('title'),
            # content may include newlines -- keep as-is, pandas will quote
            'content': raw_doc.get('content')
        }
        raw_buffer.append(raw_entry)

        # cleaning and normalization
        title = clean_text(raw_doc.get('title') or '')
        content = clean_text(raw_doc.get('content') or '')
        language = raw_doc.get('language') or safe_detect_lang(content or title)
        if language == 'unknown':
            language = safe_detect_lang((content or title)[:200] or 'en')

        sentences = []
        entities = []
        if use_ner and (content or title):
            doc_nlp = nlp(content)
            sentences = [sent.text.strip() for sent in doc_nlp.sents]
            for ent in doc_nlp.ents:
                entities.append({'record_id': raw_doc.get('record_id'),
                                 'text': ent.text,
                                 'label': ent.label_,
                                 'start': ent.start_char,
                                 'end': ent.end_char})
        else:
            # basic sentence split fallback
            sentences = re.split(r'(?<=[.!?])\s+', content) if content else []

        claim_candidates = extract_claims_simple(content)

        processed_entry = {
            'record_id': raw_doc.get('record_id'),
            'source_type': raw_doc.get('source_type'),
            'source_name': raw_doc.get('source_name'),
            'title': title,
            'content': content,
            'language': language,
            'published_at': raw_doc.get('published_at'),
            'collected_at': raw_doc.get('collected_at'),
            'text_length': raw_doc.get('text_length') or len(content),
            'num_sentences': len(sentences),
            'num_entities': len(entities),
            'claim_count': len(claim_candidates),
            'topic_labels': '|'.join(raw_doc.get('topic_labels') or []),
            '_processed_at': datetime.utcnow().isoformat()
        }

        proc_buffer.append(processed_entry)

        # normalize entities
        for ent in entities:
            entities_buffer.append({
                'record_id': ent['record_id'],
                'entity_text': ent['text'],
                'entity_label': ent['label'],
                'start': ent.get('start'),
                'end': ent.get('end'),
                '_extracted_at': datetime.utcnow().isoformat()
            })

        # normalize claims
        for claim in claim_candidates:
            claims_buffer.append({
                'record_id': raw_doc.get('record_id'),
                'claim_text': claim,
                '_extracted_at': datetime.utcnow().isoformat()
            })

        # chunked write
        if (i % chunk_size == 0) or (i == n_docs):
            # processed records
            if proc_buffer:
                df_proc = pd.DataFrame(proc_buffer)
                df_proc.to_csv(processed_csv, mode='a', header=first_proc_chunk, index=False, encoding='utf-8')
                first_proc_chunk = False
                proc_buffer = []

            # raw records
            if write_raw_csv and raw_buffer:
                df_raw = pd.DataFrame(raw_buffer)
                df_raw.to_csv(raw_csv, mode='a', header=first_raw_chunk, index=False, encoding='utf-8')
                first_raw_chunk = False
                raw_buffer = []

            # entities
            if entities_buffer:
                df_ent = pd.DataFrame(entities_buffer)
                df_ent.to_csv(entities_csv, mode='a', header=first_entities_chunk, index=False, encoding='utf-8')
                first_entities_chunk = False
                entities_buffer = []

            # claims
            if claims_buffer:
                df_claim = pd.DataFrame(claims_buffer)
                df_claim.to_csv(claims_csv, mode='a', header=first_claims_chunk, index=False, encoding='utf-8')
                first_claims_chunk = False
                claims_buffer = []

    # write metadata
    meta = {
        'dataset_version': 'v1',
        'generated_at': datetime.utcnow().isoformat(),
        'n_records': n_docs,
        'script': 'preprocess_and_ingest_to_csv.py',
        'notes': {
            'spaCy_available': SPACY_AVAILABLE,
            'used_ner': use_ner,
            'chunk_size': chunk_size
        }
    }
    with open(metadata_json, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2)

    print('Wrote processed CSVs to:', out_dir)
    print('Files:')
    print(' -', processed_csv)
    if write_raw_csv:
        print(' -', raw_csv)
    print(' -', entities_csv)
    print(' -', claims_csv)
    print(' -', metadata_json)
    print('Done.')

# -------------------------
# CLI
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess synthetic TruthLens dataset and write CSV outputs.')
    parser.add_argument('--input', type=str, default='./data/raw/synthetic_all_sources_v1.jsonl', help='Path to input JSONL')
    parser.add_argument('--outdir', type=str, default='./data/processed', help='Output directory for CSVs')
    parser.add_argument('--chunk', type=int, default=1000, help='Chunk size for writing CSVs')
    parser.add_argument('--skip-ner', action='store_true', help='Force skip spaCy NER')
    parser.add_argument('--no-raw-csv', action='store_true', help='Do not write raw_records.csv')
    parser.add_argument('--gzip', action='store_true', help='Write compressed CSVs (.gz)')
    parser.add_argument('--sample', type=int, default=None, help='Process only first N records (for testing)')
    args = parser.parse_args()

    preprocess_to_csv(input_path=args.input, out_dir=args.outdir, chunk_size=args.chunk, skip_ner=args.skip_ner, write_raw_csv=(not args.no_raw_csv), gzip=args.gzip, sample_n=args.sample)
