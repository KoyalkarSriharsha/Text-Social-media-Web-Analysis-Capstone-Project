#!/usr/bin/env python3
"""
synthetic_truthlens_generator.py

Colab/Local-ready Python script to synthesize a mixed dataset (news + GDELT + Reddit)
and save to ./data/raw/synthetic_all_sources_v1.jsonl and synthetic_all_sources_v1.parquet (optional).

Usage:
    python synthetic_truthlens_generator.py
"""
import os
import json
import random
import uuid
from datetime import datetime, timedelta
from faker import Faker
import numpy as np
import pandas as pd

FOLDER_OUT = './data/raw'
OUT_JSONL = os.path.join(FOLDER_OUT, 'synthetic_all_sources_v1.jsonl')
OUT_PARQUET = os.path.join(FOLDER_OUT, 'synthetic_all_sources_v1.parquet')
SEED = 42
N_RECORDS = 10000

SPLIT = {'newsapi': 0.5, 'gdelt': 0.15, 'reddit': 0.35}

random.seed(SEED)
np.random.seed(SEED)
Faker.seed(SEED)
fake = Faker()

os.makedirs(FOLDER_OUT, exist_ok=True)
print('Output folder:', os.path.abspath(FOLDER_OUT))

N_news = int(N_RECORDS * SPLIT['newsapi'])
N_gdelt = int(N_RECORDS * SPLIT['gdelt'])
N_reddit = N_RECORDS - N_news - N_gdelt
print('Records split -> news:', N_news, 'gdelt:', N_gdelt, 'reddit:', N_reddit)

TOPICS = ['elections','policy','economy','health','technology','sports','climate','science','international_relations','business','transport','education']
ORGANIZATIONS = ['ExampleCorp','GlobalHealthOrg','OpenResearchLab','MetroTransit','StateUniv']
PEOPLE = [fake.name() for _ in range(200)]
CITIES = ['New York','London','Mumbai','Hyderabad','Sydney','Toronto','Berlin','Beijing']
COUNTRIES = ['USA','UK','India','Australia','Canada','Germany','China']
EMOTIONS = ['joy','sadness','anger','fear','disgust','surprise','neutral']
SENTIMENT_LABELS = ['positive','neutral','negative']
CLAIM_BANK = ['Vaccines cause long-term infertility','Government plans to increase taxes by 20%','Local factory emits toxic waste into river','New drug reduces hospitalization by 50% in trials','Elections results delayed due to software glitch']

N_EVENTS = max(50, N_RECORDS // 20)
event_ids = [f'event_{i:04d}' for i in range(N_EVENTS)]
today = datetime.utcnow()
event_meta = {}
for eid in event_ids:
    center = today - timedelta(days=random.randint(0, 90), hours=random.randint(0,23))
    burst_strength = random.choice([1,2,3,5,8])
    topic = random.choice(TOPICS)
    event_meta[eid] = {'center_date': center, 'burst_strength': burst_strength, 'topic': topic,
                       'locations': [{'country': random.choice(COUNTRIES), 'city': random.choice(CITIES),
                                      'lat': round(random.uniform(-90,90),4), 'lon': round(random.uniform(-180,180),4)}]}

event_sizes = np.random.zipf(1.6, size=N_EVENTS)
norm = (event_sizes / event_sizes.sum()) * int(N_RECORDS * 0.7)
event_sizes = np.maximum(1, norm.astype(int)).tolist()
event_record_allocation = {eid:event_sizes[i] for i,eid in enumerate(event_ids)}
total_alloc = sum(event_record_allocation.values())
if total_alloc > N_RECORDS:
    scale = N_RECORDS / total_alloc
    for eid in event_record_allocation:
        event_record_allocation[eid] = max(1, int(event_record_allocation[eid] * scale))

event_pool = []
for eid,cnt in event_record_allocation.items():
    event_pool.extend([eid] * max(1, int(cnt)))
random.shuffle(event_pool)

def timestamp_near(center, window_hours=72):
    delta = timedelta(hours=random.uniform(-window_hours, window_hours))
    return center + delta

def random_paragraph(n_sentences=3):
    return ' '.join(fake.sentence(nb_words=random.randint(8,18)) for _ in range(n_sentences))

def gen_entities(num=2):
    ents = []
    for _ in range(num):
        typ = random.choice(['PERSON','ORG','GPE'])
        if typ == 'PERSON':
            text = random.choice(PEOPLE)
        elif typ == 'ORG':
            text = random.choice(ORGANIZATIONS)
        else:
            text = random.choice(CITIES)
        ents.append({'text': text, 'label': typ, 'start': None, 'end': None, 'canonical_id': None})
    return ents

def pop_event():
    if not event_pool or random.random() > 0.7:
        return None
    eid = event_pool.pop()
    meta = event_meta[eid]
    return {'event_id': eid, 'event_date': timestamp_near(meta['center_date'], window_hours=meta['burst_strength']*12), 'topic': meta['topic'], 'locations': meta['locations']}

def make_news_record(i, assigned_event=None):
    rid = f'news_{i:06d}'
    title = fake.sentence(nb_words=random.randint(5,12))
    content = random_paragraph(n_sentences=random.randint(5,12)) + '\n\n' + random_paragraph(n_sentences=random.randint(2,6))
    published_at = datetime.utcnow() - timedelta(days=random.randint(0,90), hours=random.randint(0,23))
    source_domain = random.choice(['examplenews.com','globaltimes.example','localdaily.example','techinsights.example'])
    author = fake.name()
    section = random.choice(['Politics','Business','Health','Technology','World','Sports','Science'])
    topic_labels = [assigned_event['topic']] if assigned_event else [random.choice(TOPICS)]
    emotion = random.choices(EMOTIONS, weights=[0.12,0.12,0.18,0.1,0.05,0.08,0.35], k=1)[0]
    sentiment_label = random.choices(SENTIMENT_LABELS, weights=[0.3,0.4,0.3], k=1)[0]
    credibility = round(random.random(),3)
    claim_present = random.random() < 0.12
    claim_list = []
    if claim_present:
        c = random.choice(CLAIM_BANK)
        claim_list.append({'claim_text': c, 'claim_span': None, 'claim_id': str(uuid.uuid4())})
    record = {'record_id': rid, 'source_type': 'newsapi', 'source_name': random.choice(['ExampleNews','GlobalTimes','LocalDaily','TechInsights']),
              'source_domain': source_domain, 'url': f'https://{source_domain}/article/{rid}', 'language': 'en',
              'published_at': published_at.isoformat() + 'Z', 'collected_at': datetime.utcnow().isoformat() + 'Z',
              'title': title, 'content': content, 'author': author, 'section': section, 'entities': gen_entities(num=random.randint(1,4)),
              'topic_labels': topic_labels, 'sentiment_label': sentiment_label, 'sentiment_score': round(random.uniform(-1,1),3),
              'emotion_labels': [emotion], 'factuality_label': random.choice(['True','False','Mixed','Unverifiable']),
              'credibility_score': credibility, 'claim_list': claim_list, 'is_opinion': random.random() < 0.08, 'num_images': random.randint(0,3),
              'headline_clickbait_score': round(random.random(),3), 'text_length': len(content)}
    if assigned_event:
        record['event_id'] = assigned_event['event_id']
        record['event_date'] = assigned_event['event_date'].isoformat() + 'Z'
    return record

def make_gdelt_record(i, assigned_event=None):
    rid = f'gdelt_{i:06d}'
    event_date = assigned_event['event_date'] if assigned_event else (datetime.utcnow() - timedelta(days=random.randint(0,90)))
    tone = round(random.uniform(-100,100),2)
    actors = [random.choice(PEOPLE) for _ in range(random.randint(0,3))]
    locations = assigned_event['locations'] if assigned_event else [{'country':random.choice(COUNTRIES),'city':random.choice(CITIES),'lat':round(random.uniform(-90,90),4),'lon':round(random.uniform(-180,180),4)}]
    record = {'record_id': rid, 'source_type': 'gdelt', 'source_name': 'GDELT_sim', 'gdelt_event_id': f'GDELT{random.randint(100000,999999)}',
              'event_date': event_date.isoformat() + 'Z', 'actors': actors, 'locations': locations, 'theme_codes': [assigned_event['topic']] if assigned_event else [random.choice(TOPICS)],
              'tone': tone, 'source_articles': [], 'summary_extractive': fake.sentence(nb_words=20), 'text_length': random.randint(100,400)}
    if assigned_event:
        record['event_id'] = assigned_event['event_id']
    return record

def make_reddit_record(i, assigned_event=None):
    rid = f'reddit_{i:06d}'
    subreddit = random.choice(['worldnews','news','politics','technology','science','ukpolitics','india'])
    title = fake.sentence(nb_words=random.randint(6,14))
    content = fake.paragraph(nb_sentences=random.randint(1,5))
    created_at = datetime.utcnow() - timedelta(days=random.randint(0,90), hours=random.randint(0,23))
    post_score = random.randint(-5, 5000)
    num_comments = random.randint(0,300)
    thread_comments = []
    for c in range(min(5, num_comments)):
        thread_comments.append({'comment_id': str(uuid.uuid4()), 'author': fake.user_name(), 'text': fake.sentence(nb_words=random.randint(6,20)),
                                'created_at': (created_at + timedelta(minutes=random.randint(0,1000))).isoformat() + 'Z', 'score': random.randint(-2,500)})
    topic_labels = [assigned_event['topic']] if assigned_event else [random.choice(TOPICS)]
    emotion = random.choices(EMOTIONS, weights=[0.12,0.12,0.18,0.1,0.05,0.08,0.35], k=1)[0]
    record = {'record_id': rid, 'source_type': 'reddit', 'source_name': subreddit, 'post_id': rid, 'subreddit': subreddit,
              'author_username': fake.user_name(), 'author_karma': random.randint(0,50000), 'post_score': post_score, 'num_comments': num_comments,
              'title': title, 'content': content, 'thread_comments': thread_comments, 'flair': None, 'topic_labels': topic_labels,
              'sentiment_label': random.choice(SENTIMENT_LABELS), 'emotion_labels': [emotion], 'text_length': len(content)}
    if assigned_event:
        record['event_id'] = assigned_event['event_id']
        record['event_date'] = assigned_event['event_date'].isoformat() + 'Z'
    return record

# Build records
records = []
for i in range(N_news):
    assigned = pop_event()
    rec = make_news_record(i, assigned_event=assigned)
    records.append(rec)
for i in range(N_gdelt):
    assigned = pop_event()
    rec = make_gdelt_record(i, assigned_event=assigned)
    records.append(rec)
for i in range(N_reddit):
    assigned = pop_event()
    rec = make_reddit_record(i, assigned_event=assigned)
    records.append(rec)
while len(records) < N_RECORDS:
    rec = make_news_record(len(records), assigned_event=None)
    records.append(rec)

print('Generated records:', len(records))
random.shuffle(records)

with open(OUT_JSONL, 'w', encoding='utf-8') as fh:
    for r in records:
        fh.write(json.dumps(r, ensure_ascii=False) + '\n')

print('Saved JSONL ->', os.path.abspath(OUT_JSONL))

df = pd.DataFrame(records)
try:
    df.to_parquet(OUT_PARQUET, index=False)
    print('Saved Parquet ->', os.path.abspath(OUT_PARQUET))
except Exception as e:
    print('Parquet save failed (maybe pyarrow not installed) — error:', e)

from collections import Counter
counter = Counter([r['source_type'] for r in records])
print('Counts by source_type:', counter)
print('\nSample record (first):')
print(json.dumps(records[0], indent=2, ensure_ascii=False)[:1200])
