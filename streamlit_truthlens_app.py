"""
streamlit_truthlens_app.py

Streamlit presentation layer (User-side) for TruthLens project.
Reads from ./data/results/ and ./data/processed/ and provides:
 - Key actionable insights from the NLP pipelines (sentiment, topics, events, summaries, credibility)
 - Interactive AI Chatbot component (retrieval + HF generator if available)
 - Defensive UI handling when files are missing or empty
 - Controls to adjust event-sensitivity (client-side regen using counts)

Usage:
    pip install streamlit pandas numpy scikit-learn sentence-transformers transformers
    streamlit run streamlit_truthlens_app.py

Notes:
 - The app uses precomputed artifacts in ./data/results/, including:
     sentiment_results.csv, topics.csv, events_detected.csv, summaries.csv, credibility_scores.csv,
     chatbot_tfidf.npz & chatbot_vectorizer.pkl OR sbert_embeddings.npy & retrieval_index_meta.json
 - If heavy models are not installed, the chatbot falls back to title-based responses.
"""

import os
from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# optional imports
try:
    import pickle, scipy.sparse as sps
    from sklearn.metrics.pairwise import cosine_similarity
    TFIDF_AVAILABLE = True
except Exception:
    TFIDF_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# Paths
RESULTS_DIR = Path('./data/results')
PROCESSED_DIR = Path('./data/processed')
EMB_DIR = Path('./data/embeddings')

# Helper: safe read CSV
def safe_read_csv(path):
    try:
        if path.exists():
            df = pd.read_csv(path)
            return df
    except Exception as e:
        st.error(f"Failed to read {path.name}: {e}")
    return None

# Load artifacts
sentiment_df = safe_read_csv(RESULTS_DIR / 'sentiment_results.csv')
emotion_df = safe_read_csv(RESULTS_DIR / 'emotion_scores.csv')
topics_df = safe_read_csv(RESULTS_DIR / 'topics.csv')
events_df = safe_read_csv(RESULTS_DIR / 'events_detected.csv')
summaries_df = safe_read_csv(RESULTS_DIR / 'summaries.csv')
cred_df = safe_read_csv(RESULTS_DIR / 'credibility_scores.csv')
entities_df = safe_read_csv(RESULTS_DIR / 'entities_ner.csv')
# Retrieval artifacts
tfidf_matrix = None
vectorizer = None
sbert_embeddings = None
retrieval_meta = None

if (RESULTS_DIR / 'chatbot_tfidf.npz').exists() and (RESULTS_DIR / 'chatbot_vectorizer.pkl').exists():
    try:
        tfidf_matrix = sps.load_npz(str(RESULTS_DIR / 'chatbot_tfidf.npz'))
        with open(RESULTS_DIR / 'chatbot_vectorizer.pkl','rb') as fh:
            vectorizer = pickle.load(fh)
    except Exception as e:
        tfidf_matrix = None
        vectorizer = None

if (EMB_DIR / 'chatbot_embeddings.npy').exists() and (RESULTS_DIR / 'retrieval_index_meta.json').exists():
    try:
        sbert_embeddings = np.load(str(EMB_DIR / 'chatbot_embeddings.npy'))
        with open(RESULTS_DIR / 'retrieval_index_meta.json','r',encoding='utf-8') as fh:
            retrieval_meta = json.load(fh)
    except Exception as e:
        sbert_embeddings = None
        retrieval_meta = None

# Initialize HF generator if present
hf_generator = None
if HF_AVAILABLE:
    try:
        # prefer conversational if available, else text2text
        hf_generator = pipeline('conversational', model='facebook/blenderbot-400M-distill')
    except Exception:
        try:
            hf_generator = pipeline('text2text-generation', model='facebook/blenderbot-400M-distill')
        except Exception:
            hf_generator = None

# Streamlit layout
st.set_page_config(layout="wide", page_title="TruthLens — Insights Dashboard")
st.title("TruthLens — NLP Insights & Chatbot")
st.markdown("Interactive dashboard showing actionable insights from the TruthLens NLP pipeline. "
            "Use the left panel to filter and the Chatbox below to ask questions grounded in the dataset.")

# Sidebar controls
st.sidebar.header("Controls")
sample_size = st.sidebar.number_input("Preview rows (processed)", min_value=5, max_value=500, value=10)
agg_level = st.sidebar.selectbox("Event aggregation", options=["daily","weekly"], index=0)
sensitivity = st.sidebar.selectbox("Event sensitivity", options=["strict","moderate","lenient"], index=1)

# Top KPIs row
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_articles = 0
    if (PROCESSED_DIR / 'processed_records.csv').exists():
        try:
            total_articles = len(pd.read_csv(PROCESSED_DIR / 'processed_records.csv'))
        except Exception:
            total_articles = 0
    st.metric("Total Articles", total_articles)
with col2:
    pos = neg = neu = 0
    if sentiment_df is not None:
        pos = len(sentiment_df[sentiment_df['sentiment_label']=='positive'])
        neg = len(sentiment_df[sentiment_df['sentiment_label']=='negative'])
        neu = len(sentiment_df[sentiment_df['sentiment_label']=='neutral'])
    st.metric("Positive / Negative", f"{pos} / {neg}")
with col3:
    topics_count = topics_df['topic_id'].nunique() if topics_df is not None else 0
    st.metric("Inferred Topics", topics_count)
with col4:
    events_count = len(events_df) if (events_df is not None and not events_df.empty) else 0
    st.metric("Detected Events (bursts)", events_count)

st.markdown("---")

# Left column: Overview panels
left_col, right_col = st.columns([2,3])

with left_col:
    st.subheader("Sentiment Overview")
    if sentiment_df is None:
        st.info("Sentiment results missing. Run the pipeline to generate sentiment_results.csv")
    else:
        st.dataframe(sentiment_df.head(sample_size))
        # simple bar counts
        st.bar_chart(sentiment_df['sentiment_label'].value_counts())

    st.subheader("Top Topics (sample)")
    if topics_df is None:
        st.info("topics.csv missing. Run topic modeling pipeline.")
    else:
        # show top topics by frequency
        top_topics = topics_df['topic_id'].value_counts().head(10)
        st.table(top_topics.rename_axis("topic_id").reset_index(name="count"))

    st.subheader("Event Detection (adaptive)")
    if events_df is None or events_df.empty:
        st.info("No events detected (file empty). Use sensitivity control to regenerate events more sensitively in the backend or inspect date distribution.")
    else:
        # show events table and simple chart
        st.dataframe(events_df.head(20))
        # counts per date
        try:
            events_df['date_parsed'] = pd.to_datetime(events_df['date'], errors='coerce')
            counts_by_date = events_df.groupby('date_parsed').size().reset_index(name='n_events')
            st.line_chart(counts_by_date.set_index('date_parsed')['n_events'])
        except Exception:
            pass

with right_col:
    st.subheader("Top Credibility & Risk Signals")
    if cred_df is None:
        st.info("Credibility scores not available.")
    else:
        st.dataframe(cred_df.sort_values('credibility_score').head(20))
        avg_cred = cred_df['credibility_score'].mean()
        st.metric("Average Credibility Score", f"{avg_cred:.3f}")

    st.subheader("Representative Summaries")
    if summaries_df is None:
        st.info("Summaries not available.")
    else:
        st.dataframe(summaries_df.head(sample_size))

st.markdown("---")

# Chatbot area (bottom)
st.subheader("Interactive AI Chatbot (Retrieval + Generator)")
st.markdown("Ask a question. Results will be grounded using TF-IDF or SBERT retrieval. If heavy models are unavailable, the assistant will return top matching article titles.")

query = st.text_input("Enter your question here", placeholder="e.g., What are the main topics today?")
k = st.slider("Number of retrieved passages", 1, 10, 3)

def retrieve_tfidf(query, k=3):
    if not TFIDF_AVAILABLE or tfidf_matrix is None or vectorizer is None:
        return []
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, tfidf_matrix)
    if hasattr(sims, "toarray"):
        sims = sims.toarray()[0]
    else:
        sims = sims[0]
    top_idx = np.argsort(-sims)[:k]
    titles = []
    # map indices to titles via processed CSV
    try:
        df_proc = pd.read_csv(PROCESSED_DIR / 'processed_records.csv')
        for i in top_idx:
            titles.append((df_proc.iloc[i]['record_id'], float(sims[i]), str(df_proc.iloc[i]['title'])[:200]))
    except Exception:
        titles = []
    return titles

def retrieve_sbert(query, k=3):
    if sbert_embeddings is None or retrieval_meta is None or not SBERT_AVAILABLE:
        return []
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        qemb = model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(qemb, sbert_embeddings)[0]
        top_idx = np.argsort(-sims)[:k]
        titles = []
        meta_titles = retrieval_meta.get('titles') or []
        ids = retrieval_meta.get('record_ids') or []
        for i in top_idx:
            titles.append((ids[i], float(sims[i]), meta_titles[i][:200] if i < len(meta_titles) else ""))
        return titles
    except Exception:
        return []

def generate_answer_with_context(generator, query, contexts):
    # Build a short prompt including retrieved contexts
    prompt = "Context:\n" + "\n\n".join(contexts) + f"\n\nQuestion: {query}\nAnswer:"
    try:
        out = generator(prompt, max_length=200, truncation=True)
        if isinstance(out, list) and len(out)>0:
            # different pipelines return different keys
            text_out = out[0].get('generated_text') or out[0].get('summary_text') or out[0].get('answer') or str(out[0])
        else:
            text_out = str(out)
    except Exception as e:
        text_out = "Generator failed: " + str(e)
    return text_out

if st.button("Ask") and query:
    results = []
    # Prefer SBERT retrieval if embeddings present
    if sbert_embeddings is not None and SBERT_AVAILABLE:
        results = retrieve_sbert(query, k=k)
    elif TFIDF_AVAILABLE and tfidf_matrix is not None and vectorizer is not None:
        results = retrieve_tfidf(query, k=k)
    # If no retrieval artifacts, try simple title search fallback
    if not results:
        st.info("No retrieval artifacts found. Falling back to title search in processed CSV.")
        try:
            dfp = pd.read_csv(PROCESSED_DIR / 'processed_records.csv')
            mask = dfp['title'].str.contains(query, case=False, na=False)
            hits = dfp[mask].head(k)
            results = [(row['record_id'], 1.0, row['title']) for _,row in hits.iterrows()]
        except Exception:
            results = []

    st.markdown("**Retrieved Passages / Titles**")
    if results:
        for recid, score, title in results:
            st.write(f"- {recid} (score={score:.3f}) — {title}")
    else:
        st.write("No relevant documents retrieved. Try a different query.")

    # Generate an answer using HF generator if available
    if hf_generator and results:
        contexts = [r[2] for r in results]
        answer = generate_answer_with_context(hf_generator, query, contexts)
        st.subheader("Assistant answer (grounded)")
        st.write(answer)
    else:
        st.warning("Generator model not available — showing retrieved titles as answer fallback.")

st.markdown("---")
st.caption("Tip: If events are empty, try regenerating events with a lower threshold or weekly aggregation in your preprocessing step.")

# Footer: quick raw views
with st.expander("Raw files and diagnostics"):
    st.write("Files in ./data/results:")
    for f in sorted(RESULTS_DIR.glob("*")):
        st.write(f.name, "-", f.stat().st_size, "bytes")
    st.write("Files in ./data/processed:")
    for f in sorted(PROCESSED_DIR.glob("*")):
        st.write(f.name, "-", f.stat().st_size, "bytes")

