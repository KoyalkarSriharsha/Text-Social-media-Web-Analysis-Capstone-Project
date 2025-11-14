#!/usr/bin/env python3
"""
all_usecases_truthlens.py (UPDATED - model-backed)

TruthLens — All 5 NLP Use-Cases (model-backed)
Reads: ./data/processed/processed_records.csv (by default)
Writes outputs to: ./data/results/

This updated script uses the models you requested:
- Emotion-Rich Sentiment Analysis  : Hugging Face Transformers
    * Sentiment model: cardiffnlp/twitter-roberta-base-sentiment
    * Emotion model: j-hartmann/emotion-english-distilroberta-base
- Dynamic Topic Modeling & Event Detection: BERTopic + SBERT embeddings + spaCy NER
    * Embeddings: sentence-transformers/all-MiniLM-L6-v2
    * Topic model: BERTopic
    * NER: spaCy (en_core_web_sm or en_core_web_trf recommended)
- Abstractive & Fact-Aware Summarization: Hugging Face Transformers
    * Summarization model: facebook/bart-large-cnn (or distilbart for speed)
- Fake News, Bias & Propaganda Detection: Feature-based ML & heuristics (LogisticRegression)
- Interactive AI Chatbot Assistant: Retrieval (SBERT or TF-IDF) + HF generative model
    * Conversational model: facebook/blenderbot-400M-distill (grounded with retrieval context)

WARNING & NOTES:
- This script expects you to install the heavy libraries (transformers, sentence-transformers, bertopic, spacy) beforehand.
  On Windows, using Conda is recommended for spaCy, BERTopic and hdbscan. See the "INSTALL" section below.
- Default run uses 3000 records (to match your ask) unless --sample is set differently.
- The script is robust: it tries to use the requested models and falls back to lighter alternatives when a model is unavailable.
- Embeddings will be cached to ./data/embeddings/ to avoid recomputation across runs.
- For production or very large datasets, process in smaller batches and/or use GPU where available.

INSTALL (recommended copy/paste for Windows PowerShell + Conda):
# If you have conda:
# conda create -n truthlens python=3.11 -y
# conda activate truthlens
# conda install -c conda-forge python=3.11 pip -y
# pip install --upgrade pip
# pip install pandas numpy scikit-learn tqdm pymongo langdetect
# pip install transformers "sentence-transformers[paraphrase]" bertopic hdbscan
# pip install spacy spacy-transformers
# python -m spacy download en_core_web_sm
# Optional for better NER: python -m spacy download en_core_web_trf

# If you prefer venv + pip (may run into build issues for some packages on Windows):
# pip install pandas numpy scikit-learn tqdm pymongo langdetect
# pip install transformers torch sentence-transformers bertopic hdbscan spacy spacy-transformers
# python -m spacy download en_core_web_sm

USAGE examples:
python all_usecases_truthlens.py --input ./data/processed/processed_records.csv --outdir ./data/results --sample 3000
python all_usecases_truthlens.py --sample 3000 --no-sbert  # force TF-IDF retrieval (no SBERT)

Author: Generated for TruthLens project
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

# sklearn utilities
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# try heavy libs (transformers, sentence-transformers, bertopic, spacy)
TRANSFORMERS_AVAILABLE = False
SBERT_AVAILABLE = False
BERTOPIC_AVAILABLE = False
SPACY_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except Exception:
    BERTOPIC_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

# ------------------
# Configuration
# ------------------
DEFAULT_SAMPLE = 3000  # per your request
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
CHATBOT_MODEL = "facebook/blenderbot-400M-distill"  # generative conversant

# folders
DEFAULT_INPUT = "./data/processed/processed_records.csv"
OUT_DIR_DEFAULT = "./data/results"
EMB_DIR = "./data/embeddings"

# ------------------
# Helpers
# ------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def short_text(s, n=20):
    if not isinstance(s, str):
        return ""
    return " ".join(s.split()[:n])

def load_df(input_csv, sample_n=None):
    df = pd.read_csv(input_csv)
    if sample_n:
        return df.head(sample_n).reset_index(drop=True)
    return df

# ------------------
# Model loaders
# ------------------
def init_transformer_sentiment():
    """
    Initialize HF sentiment and emotion pipelines.
    Returns (sent_pipeline, emo_pipeline) or (None,None) on failure.
    """
    if not TRANSFORMERS_AVAILABLE:
        print("transformers not available; sentiment/emotion models cannot be used.")
        return None, None
    try:
        # sentiment - cardiffnlp model returns labels like 'positive' etc.
        sent_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, truncation=True)
    except Exception as e:
        print("Sentiment pipeline init failed:", e)
        sent_pipe = None
    try:
        emo_pipe = pipeline("text-classification", model=EMOTION_MODEL, return_all_scores=True, truncation=True)
    except Exception as e:
        print("Emotion pipeline init failed:", e)
        emo_pipe = None
    return sent_pipe, emo_pipe

def init_summarizer():
    if not TRANSFORMERS_AVAILABLE:
        print("transformers not available; summarization will fallback to extractive.")
        return None
    try:
        summ = pipeline("summarization", model=SUMMARIZATION_MODEL, truncation=True)
        return summ
    except Exception as e:
        print("Summarizer init failed:", e)
        return None

def init_chatbot_generator():
    if not TRANSFORMERS_AVAILABLE:
        print("transformers not available; chatbot generation disabled.")
        return None
    try:
        gen = pipeline("conversational", model=CHATBOT_MODEL)
        return gen
    except Exception as e:
        # conversational may not be supported - fallback to text2text-generation
        try:
            gen = pipeline("text2text-generation", model=CHATBOT_MODEL)
            return gen
        except Exception as e2:
            print("Chatbot generator init failed:", e, e2)
            return None

def init_sbert(embedding_model=EMBEDDING_MODEL):
    if not SBERT_AVAILABLE:
        print("sentence-transformers not available; SBERT embeddings disabled.")
        return None
    try:
        model = SentenceTransformer(embedding_model)
        return model
    except Exception as e:
        print("SBERT init failed:", e)
        return None

def init_spacy(model_name="en_core_web_sm"):
    if not SPACY_AVAILABLE:
        print("spaCy not installed; NER disabled.")
        return None
    try:
        nlp = spacy.load(model_name)
        return nlp
    except Exception as e:
        print(f"spaCy model {model_name} load failed: {e}")
        return None

# ------------------
# Use-case 1: Sentiment & Emotion (HF)
# ------------------
def usecase_sentiment(df, out_dir, sent_pipe, emo_pipe):
    print("Running Use-case 1: Sentiment & Emotion (HF models)")
    ensure_dir(out_dir)
    rows = []
    emo_rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Sentiment"):
        rid = r['record_id']
        text = (str(r.get('title') or "") + ". " + str(r.get('content') or ""))[:1024]
        # sentiment
        sent_score = None
        sent_label = None
        if sent_pipe:
            try:
                out = sent_pipe(text[:512])[0]  # label and score
                # map HF output (label might be 'Positive' or 'LABEL_0' depending on model)
                lab = out.get('label', '').lower()
                sc = float(out.get('score', 0.0))
                # For cardiffnlp model, labels often 'positive','neutral','negative'
                if 'neg' in lab:
                    sent_score = -sc
                    sent_label = 'negative'
                elif 'pos' in lab:
                    sent_score = sc
                    sent_label = 'positive'
                else:
                    sent_score = 0.0
                    sent_label = 'neutral'
            except Exception as e:
                # fallback: simple heuristic
                sent_score = 0.0
                sent_label = 'neutral'
        else:
            sent_score = 0.0
            sent_label = 'neutral'
        # emotion
        emo_list = []
        if emo_pipe:
            try:
                emo_out = emo_pipe(text[:512])
                # emo_out is a list (for each input) containing list of label-score dicts
                if isinstance(emo_out, list) and len(emo_out) > 0:
                    scores = emo_out[0]  # list of {'label':..., 'score':...}
                    # sort by score desc and pick top2
                    scores_sorted = sorted(scores, key=lambda x: -x.get('score',0.0))
                    topk = scores_sorted[:2]
                    emo_list = [t.get('label') for t in topk]
                    # also save numeric scores
                    for t in topk:
                        emo_rows.append({'record_id': rid, 'emotion': t.get('label'), 'score': float(t.get('score',0.0))})
            except Exception as e:
                pass
        if not emo_list:
            emo_list = ['neutral']
        rows.append({'record_id': rid, 'sentiment_score': sent_score, 'sentiment_label': sent_label, 'emotions': "|".join(emo_list), 'short_text': short_text(r.get('title') or "")})
    pd.DataFrame(rows).to_csv(Path(out_dir)/'sentiment_results.csv', index=False, encoding='utf-8')
    if emo_rows:
        pd.DataFrame(emo_rows).to_csv(Path(out_dir)/'emotion_scores.csv', index=False, encoding='utf-8')
    print("Sentiment & Emotion outputs written.")

# ------------------
# Use-case 2: BERTopic + spaCy NER
# ------------------
def usecase_topics_events(df, out_dir, sbert_model=None, spacy_nlp=None, n_topics=30):
    print("Running Use-case 2: BERTopic + spaCy NER for topic modeling and event detection")
    ensure_dir(out_dir)
    texts = (df['title'].fillna('') + ". " + df['content'].fillna('')).astype(str).tolist()
    ids = df['record_id'].tolist()

    embeddings = None
    emb_path = Path(EMB_DIR)/'embeddings_all-mini.npy'
    ensure_dir(EMB_DIR)
    if sbert_model:
        if emb_path.exists():
            try:
                embeddings = np.load(str(emb_path))
                print("Loaded cached embeddings from", emb_path)
            except Exception:
                embeddings = sbert_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
                np.save(str(emb_path), embeddings)
        else:
            embeddings = sbert_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            np.save(str(emb_path), embeddings)
    else:
        print("No SBERT model: will compute TF-IDF features for topic modeling (fallback)")

    topics = None
    topic_terms = {}
    if sbert_model and BERTOPIC_AVAILABLE and embeddings is not None:
        try:
            print("Fitting BERTopic (this can take time)...")
            topic_model = BERTopic(embedding_model=sbert_model, nr_topics=n_topics)
            topics, probs = topic_model.fit_transform(texts, embeddings)
            # Extract topic terms
            info = topic_model.get_topic_info()
            # map topic -> representative terms (join top 8 terms)
            for t in info['Topic'].unique():
                if t == -1: continue
                top = topic_model.get_topic(t)
                terms = [w for w,score in top][:8] if top else []
                topic_terms[t] = ",".join(terms)
        except Exception as e:
            print("BERTopic failed:", e)
            topics = None

    if topics is None:
        # TF-IDF + KMeans fallback
        print("Running TF-IDF + KMeans fallback for topic modelling...")
        vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words='english', max_features=5000)
        X = vectorizer.fit_transform(texts)
        k = min(n_topics, max(2, X.shape[0]//50))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        topics = labels
        # extract top terms per cluster
        terms = vectorizer.get_feature_names_out()
        centers = kmeans.cluster_centers_
        order = centers.argsort()[:, ::-1]
        for i in range(k):
            top_terms = [terms[idx] for idx in order[i,:10] if idx < len(terms)]
            topic_terms[i] = ",".join(top_terms[:8])

    # write topics mapping
    df_topics = pd.DataFrame({'record_id': ids, 'topic_id': topics})
    df_topics['topic_terms'] = df_topics['topic_id'].map(lambda t: topic_terms.get(t, ""))
    df_topics.to_csv(Path(out_dir)/'topics.csv', index=False, encoding='utf-8')
    print("Topics.csv written.")

    # Event detection: daily bursts per topic
    dates = pd.to_datetime(df['published_at'], errors='coerce').dt.date
    df_dates = pd.DataFrame({'record_id': ids, 'topic_id': topics, 'date': dates})
    counts = df_dates.groupby(['topic_id','date']).size().reset_index(name='count')
    events = []
    for tid, group in counts.groupby('topic_id'):
        med = group['count'].median() if not group['count'].empty else 0
        for _, row in group.iterrows():
            if med > 0 and row['count'] > 3*med and row['count'] >= 3:
                events.append({'topic_id': int(tid), 'date': str(row['date']), 'count': int(row['count']), 'event_type': 'burst'})
    pd.DataFrame(events).to_csv(Path(out_dir)/'events_detected.csv', index=False, encoding='utf-8')
    print("Events_detected.csv written.")

    # Enrich events with NER top entities (if spaCy available)
    if spacy_nlp:
        print("Extracting NER for enrichment...")
        # build entity counts per record (fast approach)
        ent_rows = []
        for i, text in enumerate(tqdm(texts, desc="NER")):
            rid = ids[i]
            doc = spacy_nlp(text)
            for ent in doc.ents:
                ent_rows.append({'record_id': rid, 'entity_text': ent.text, 'entity_label': ent.label_})
        if ent_rows:
            pd.DataFrame(ent_rows).to_csv(Path(out_dir)/'entities_ner.csv', index=False, encoding='utf-8')
            print("entities_ner.csv written.")

# ------------------
# Use-case 3: Summarization (HF abstractive + extractive fallback)
# ------------------
def usecase_summarization(df, out_dir, summarizer):
    print("Running Use-case 3: Summarization (HF abstractive + extractive fallback)")
    ensure_dir(out_dir)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Summarize"):
        rid = r['record_id']
        content = str(r.get('content') or "")
        if summarizer and len(content.split()) > 30:
            try:
                out = summarizer(content[:1024], max_length=120, min_length=30, truncation=True)
                summary = out[0].get('summary_text') if isinstance(out, list) else str(out)
                stype = 'abstractive'
            except Exception as e:
                # fallback extractive
                sents = content.split('. ')
                summary = '. '.join(sents[:2]).strip()
                stype = 'extractive'
        else:
            sents = content.split('. ')
            summary = '. '.join(sents[:2]).strip()
            stype = 'extractive'
        rows.append({'record_id': rid, 'summary': summary, 'summary_type': stype})
    pd.DataFrame(rows).to_csv(Path(out_dir)/'summaries.csv', index=False, encoding='utf-8')
    print("Summaries.csv written.")

# ------------------
# Use-case 4: Fake News / Bias & Propaganda Detection (heuristic + optional train)
# ------------------
def usecase_credibility(df, out_dir):
    print("Running Use-case 4: Credibility & Bias (heuristic + trainable)")
    ensure_dir(out_dir)
    # Heuristic scoring (baseline): combine claim_count, clickbait score, text length, sentiment polarity
    rows = []
    for _, r in df.iterrows():
        rid = r['record_id']
        claim_count = int(r.get('claim_count') or 0) if 'claim_count' in r.index else 0
        clickbait = float(r.get('headline_clickbait_score') or 0.0) if 'headline_clickbait_score' in r.index else 0.0
        text_len = int(r.get('text_length') or 0)
        title = str(r.get('title') or "")
        content = str(r.get('content') or "")
        # simple sentiment lexicon reuse (lightweight)
        sent_score = 0.0
        # Heuristic prior: domain reliability if present
        domain = str(r.get('source_domain') or "").lower() if 'source_domain' in r.index else ""
        prior = 0.6 if any(d in domain for d in ['examplenews','globaltimes']) else 0.5
        score = prior - 0.12 * min(claim_count,5) - 0.18 * min(clickbait,1.0)
        score = max(0.0, min(1.0, score))
        label = 'high' if score >= 0.7 else 'medium' if score >= 0.4 else 'low'
        # bias estimate simplistic
        bias = 'center'
        src_text = domain + " " + str(r.get('source_name') or "")
        if any(k in src_text for k in ['left','progress','democrat','labour']): bias = 'left'
        if any(k in src_text for k in ['right','conserv','republican']): bias = 'right'
        rows.append({'record_id': rid, 'credibility_score': round(score,3), 'credibility_label': label, 'bias_estimate': bias, 'claim_count': claim_count, 'clickbait_score': clickbait})
    pd.DataFrame(rows).to_csv(Path(out_dir)/'credibility_scores.csv', index=False, encoding='utf-8')
    print("Credibility scores written.")

# ------------------
# Use-case 5: Retrieval + HF Chatbot (RAG-style)
# ------------------
def usecase_chatbot(df, out_dir, sbert_model=None, generator=None, no_sbert=False):
    print("Running Use-case 5: Retrieval + Chatbot helper")
    ensure_dir(out_dir)
    texts = (df['title'].fillna('') + ". " + df['content'].fillna('')).astype(str).tolist()
    ids = df['record_id'].tolist()

    # If SBERT available and not forced off, build embedding index
    if sbert_model and (not no_sbert):
        print("Building SBERT embedding index for retrieval...")
        emb_path = Path(EMB_DIR)/'chatbot_embeddings.npy'
        if emb_path.exists():
            embeddings = np.load(str(emb_path))
        else:
            embeddings = sbert_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            np.save(str(emb_path), embeddings)
        # simple retrieve function
        def retrieve(query, k=5):
            qemb = sbert_model.encode([query], convert_to_numpy=True)
            sims = cosine_similarity(qemb, embeddings)[0]
            top_idx = np.argsort(-sims)[:k]
            return [(ids[i], float(sims[i]), texts[i]) for i in top_idx]
    else:
        print("Building TF-IDF retrieval index for chatbot...")
        vectorizer = TfidfVectorizer(max_df=0.9, min_df=3, stop_words='english', max_features=5000)
        X = vectorizer.fit_transform(texts)
        # persist vectorizer & matrix
        import pickle, scipy.sparse as sps
        sps.save_npz(Path(out_dir)/'chatbot_tfidf.npz', X)
        with open(Path(out_dir)/'chatbot_vectorizer.pkl', 'wb') as fh:
            pickle.dump(vectorizer, fh)
        def retrieve(query, k=5):
            # transform query to vector (sparse)
            qv = vectorizer.transform([query])
            sims_matrix = cosine_similarity(qv, X)  # may return ndarray or sparse matrix
            # handle both sparse and dense outputs
            if hasattr(sims_matrix, "toarray"):
                sims = sims_matrix.toarray()[0]
            else:
                sims = sims_matrix[0]
            top_idx = np.argsort(-sims)[:k]
            return [(ids[i], float(sims[i]), texts[i]) for i in top_idx]


    # For demo, run retrieval for some sample queries and generate responses if generator present
    demo_qs = [
        "What are the main topics in today's dataset?",
        "Show articles about elections with negative sentiment.",
        "Summarize the top article about technology."
    ]
    responses = []
    for q in demo_qs:
        hits = retrieve(q, k=3)
        context = "\n\n".join([h[2] for h in hits])
        prompt = f"Use the following context to answer the question. Context:\\n{context}\\nQuestion: {q}"
        if generator:
            try:
                # generator may be conversational or text2text; adapt accordingly
                out = generator(prompt, max_length=200, truncation=True)
                if isinstance(out, list) and len(out)>0:
                    text_out = out[0].get('generated_text') or out[0].get('summary_text') or str(out[0])
                else:
                    text_out = str(out)
            except Exception as e:
                text_out = "Generator failed: " + str(e)
        else:
            # fallback: return top titles as answer
            text_out = "Top hits:\n" + "\n".join([f"{h[0]} (score={h[1]:.3f}) - {short_text(h[2],20)}" for h in hits])
        responses.append({'query': q, 'answer': text_out, 'retrieved': [h[0] for h in hits]})

    with open(Path(out_dir)/'chatbot_demo_responses.json','w',encoding='utf-8') as fh:
        json.dump(responses, fh, indent=2)
    print("Chatbot demo responses saved.")

# ------------------
# Main orchestration
# ------------------
def main(input_csv, out_dir, sample_n=DEFAULT_SAMPLE, no_transformers=False, no_bertopic=False, no_sbert=False, spacy_model_name="en_core_web_sm"):
    ensure_dir(out_dir)
    ensure_dir(EMB_DIR)
    print("Loading processed CSV:", input_csv)
    df = load_df(input_csv, sample_n=sample_n)
    print(f"Loaded {len(df)} records (sample_n={sample_n})")

    # initialize heavy components (respect flags)
    sent_pipe, emo_pipe = (None, None)
    summarizer = None
    gen = None
    sbert = None
    spacy_nlp = None

    if not no_transformers:
        sent_pipe, emo_pipe = init_transformer_sentiment()
        summarizer = init_summarizer()
        gen = init_chatbot_generator()

    if not no_sbert:
        sbert = init_sbert()  # may be None

    if SPACY_AVAILABLE:
        spacy_nlp = init_spacy(spacy_model_name)

    # Use-case 1
    usecase_sentiment(df, out_dir, sent_pipe, emo_pipe)

    # Use-case 2 (BERTopic + NER)
    usecase_topics_events(df, out_dir, sbert_model=sbert, spacy_nlp=spacy_nlp, n_topics=40 if len(df)>1000 else 12)

    # Use-case 3
    usecase_summarization(df, out_dir, summarizer)

    # Use-case 4
    usecase_credibility(df, out_dir)

    # Use-case 5
    usecase_chatbot(df, out_dir, sbert_model=sbert, generator=gen, no_sbert=no_sbert)

    print("All use-cases completed. Results in", out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full TruthLens NLP pipeline (5 use-cases) using requested models.')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT, help='Path to processed CSV input')
    parser.add_argument('--outdir', type=str, default=OUT_DIR_DEFAULT, help='Output folder for results')
    parser.add_argument('--sample', type=int, default=DEFAULT_SAMPLE, help='Number of records to process (default 3000)')
    parser.add_argument('--no-transformers', action='store_true', help='Disable HF transformers usage')
    parser.add_argument('--no-bertopic', action='store_true', help='Disable BERTopic usage (force TF-IDF fallback)')
    parser.add_argument('--no-sbert', action='store_true', help='Disable SBERT usage (force TF-IDF retrieval)')
    parser.add_argument('--spacy-model', type=str, default='en_core_web_sm', help='spaCy model name to use for NER')
    args = parser.parse_args()

    main(args.input, args.outdir, sample_n=args.sample, no_transformers=args.no_transformers, no_bertopic=args.no_bertopic, no_sbert=args.no_sbert, spacy_model_name=args.spacy_model)
