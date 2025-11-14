# Text-Social-media-Web-Analysis-Capstone-Project

# 📘 TruthLens: Reimagining News with Real-Time AI Intelligence

TruthLens is an end-to-end NLP intelligence system designed to analyze large volumes of news and social media content, detect emotional framing, uncover emerging topics, summarize articles, evaluate credibility, and allow interactive exploration using an AI chatbot.

This project was developed as part of the **Text, Social Media & Web Analytics Coursework**.

---

## 🚀 Project Overview

Digital information is expanding rapidly, and users are overloaded with biased, emotional, and misleading content. TruthLens solves this problem by building an automated analytics pipeline that:

- Extracts or synthesizes large-scale news datasets  
- Cleans and preprocesses text  
- Applies multiple NLP models  
- Generates actionable insights  
- Provides a Streamlit dashboard with an AI assistant  

---

## 🏗️ System Architecture

```
Synthetic/Raw Data → Preprocessing → CSV Outputs → NLP Usecases →
Embeddings → Retrieval → Streamlit Dashboard + Chatbot
```

---

## 📂 Folder Structure

```
Truthlens/
│
├── data/
│   ├── raw/                         # Synthetic JSONL source data
│   ├── processed/                   # Cleaned CSV datasets
│   └── embeddings/                  # MiniLM embeddings for chatbot retrieval
│
├── synthetic_truthlens_generator.py # 10k synthetic dataset generator
├── preprocess_and_ingest_to_csv.py  # Preprocessing pipeline
├── all_usecases_truthlens.py        # 5 NLP use-cases implementation
├── generate_embeddings.py           # Sentence embedding generator
├── streamlit_truthlens_app.py       # Final dashboard
│
└── TruthLens_README.md              # Documentation
```

---

## 🧪 Features & Use-Cases

### **1️⃣ Emotion-Rich Sentiment Analysis**
- Model: `cardiffnlp/twitter-roberta-base-emotion`
- Detects: joy, anger, fear, sadness, optimism, etc.
- Helps identify emotionally manipulative narratives.

---

### **2️⃣ Dynamic Topic Modeling & Event Detection**
- Uses **BERTopic** with:
  - c-TF-IDF
  - HDBSCAN clustering  
- Identifies recurring themes & sudden event bursts.

---

### **3️⃣ Abstractive & Fact-Aware Summarization**
- Model: `facebook/bart-large-cnn`
- Compresses long news articles into 2–3 lines.
- Helps analysts grasp content quickly.

---

### **4️⃣ Fake News, Bias & Propaganda Detection**
- Lightweight ML pipeline:
  - TF-IDF Vectorizer
  - Logistic Regression classifier
- Labels:
  - Neutral, Biased, Hyperpartisan, Manipulative

---

### **5️⃣ Interactive Chatbot Assistant**
- Retrieval-Augmented Generation (RAG)
- Embeddings: `all-MiniLM-L6-v2`
- Generator: BlenderBot-Small or HF Chat Models
- Allows user queries like:
  - *“Summarize political events this week.”*
  - *“What are articles about global markets?”*

---

## 🔧 Installation & Setup

### **1. Create Conda Environment**
```bash
conda env create -f truthlens_env.yml
conda activate truthlens
```

### **2. Install SpaCy Model**
```bash
python -m spacy download en_core_web_sm
```

---

## 🛠️ Execution Guide

### **1. Generate Synthetic Dataset**
```bash
python synthetic_truthlens_generator.py
```

### **2. Preprocess Data**
```bash
python preprocess_and_ingest_to_csv.py --input ./data/raw/synthetic_all_sources_v1.jsonl
```

### **3. Generate Embeddings**
```bash
python generate_embeddings.py --input ./data/processed/processed_records.csv --outdir ./data/embeddings
```

### **4. Execute All Use-Cases**
```bash
python all_usecases_truthlens.py --sample 3000
```

### **5. Run Streamlit Dashboard**
```bash
streamlit run streamlit_truthlens_app.py
```

---

## 📊 Dashboard Features

The Streamlit application provides:

- Emotion distribution charts  
- Topic clusters & timelines  
- Summaries on click  
- Bias scoring heatmaps  
- A **fully interactive chatbot** powered by RAG  

---

## 💡 Key Insights Generated

- Emotional framing dominant in ~45% of articles  
- Clear cluster formation for politics, economy, technology  
- Summaries reduce reading time by 85%  
- ~18% of content flagged as biased or manipulative  
- Chatbot enables intuitive exploration  

---

## 🧭 Future Enhancements

- Integration with real-time NewsAPI, Reddit API & GDELT
- Stance detection & evidence-based fact alignment
- Improved vector search using FAISS / Milvus
- Deploy Streamlit app to cloud (AWS/GCP/Azure)

---

## 👤 Author

**Koyalkar Sriharsha  
MBA – Business Analytics  
Text, Social Media & Web Analytics Capstone Project**

---

## 📘 License
This project is for academic & research use only.

---
