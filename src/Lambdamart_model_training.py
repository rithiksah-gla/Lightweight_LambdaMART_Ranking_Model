import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import lightgbm as lgb
import nltk
import re

# Load your dataset
df = pd.read_csv("train_dataset_lambdamart.csv")  # Ensure columns: query, candidate, score

# Initialize stemmer
stemmer = PorterStemmer()

def tokenize(text):
    # Tokenize, lowercase, stem, and remove punctuation tokens
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [stemmer.stem(token) for token in tokens]

# Step 1: Compute IDF from combined stemmed query + doc corpus
all_texts = df["query"].tolist() + df["candidate"].tolist()
all_texts_stemmed = [" ".join(tokenize(text)) for text in all_texts]

vectorizer = TfidfVectorizer()
vectorizer.fit(all_texts_stemmed)
idf_dict = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

# Step 2: Extract handcrafted features
def extract_features(query, doc, idf_dict):
    q_tokens = tokenize(query)
    d_tokens = tokenize(doc)

    num_q_terms = len(q_tokens)
    num_q_unique = len(set(q_tokens))
    num_d_terms = len(d_tokens)
    num_d_unique = len(set(d_tokens))

    q_idfs = [idf_dict.get(t, 0) for t in q_tokens]
    d_idfs = [idf_dict.get(t, 0) for t in d_tokens]

    min_q_idf = min(q_idfs) if q_idfs else 0
    max_q_idf = max(q_idfs) if q_idfs else 0
    sum_q_idf = sum(q_idfs)

    min_d_idf = min(d_idfs) if d_idfs else 0
    max_d_idf = max(d_idfs) if d_idfs else 0
    sum_d_idf = sum(d_idfs)

    overlap = len(set(q_tokens) & set(d_tokens))
    bm25_score = sum([idf_dict.get(t, 0) for t in set(q_tokens) & set(d_tokens)])

    return [
        num_q_terms, num_q_unique, num_d_terms, num_d_unique,
        min_q_idf, max_q_idf, sum_q_idf,
        min_d_idf, max_d_idf, sum_d_idf,
        overlap, bm25_score
    ]

# Define feature names
feature_names = [
    'num_q_terms', 'num_q_unique', 'num_d_terms', 'num_d_unique',
    'min_q_idf', 'max_q_idf', 'sum_q_idf',
    'min_d_idf', 'max_d_idf', 'sum_d_idf',
    'overlap', 'bm25_score'
]

# Extract features
features = df.apply(lambda row: extract_features(row['query'], row['candidate'], idf_dict), axis=1, result_type='expand')
features.columns = feature_names
df = pd.concat([df, features], axis=1)

# Step 3: Assign relevance scores (higher model score = more relevant)
df = df.sort_values(by=['query', 'score'], ascending=[True, False])
df['relevance'] = df.groupby('query').cumcount(ascending=False)

# Step 4: Prepare data for LightGBM LambdaMART
X = df[feature_names]
y = df['relevance']
group = df.groupby('query').size().to_list()

train_data = lgb.Dataset(X, label=y, group=group)

# Step 5: Train the LambdaMART model
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5],
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbosity': -1
}

model = lgb.train(params, train_data, num_boost_round=100)

# Save trained model
model.save_model("lambdamart_model.txt")

# Print feature importance
# importance = model.feature_importance(importance_type='gain')
# for name, score in zip(feature_names, importance):
#     print(f"{name}: {score:.2f}")
