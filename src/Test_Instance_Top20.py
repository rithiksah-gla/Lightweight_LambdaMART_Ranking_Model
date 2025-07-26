import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import re
from tqdm import tqdm

train_df = pd.read_csv("train.tsv", sep="\t", names=["candidate", "candidate_label"])
test_df = pd.read_csv("test.tsv", sep="\t", names=["query", "query_label"])

# Preprocessing
stemmer = PorterStemmer()

def tokenize(text):
    tokens = re.findall(r'\b\w+\b', text.lower())  # Alphanumeric words only
    return [stemmer.stem(token) for token in tokens]

# Build IDF Dictionary
all_texts = train_df["candidate"].tolist()
all_texts_stemmed = [" ".join(tokenize(text)) for text in all_texts]
vectorizer = TfidfVectorizer()
vectorizer.fit(all_texts_stemmed)
idf_dict = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

# Feature Extraction
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

    overlap = set(q_tokens) & set(d_tokens)
    bm25_score = sum([idf_dict.get(t, 0) for t in overlap])

    return [
        num_q_terms, num_q_unique, num_d_terms, num_d_unique,
        min_q_idf, max_q_idf, sum_q_idf,
        min_d_idf, max_d_idf, sum_d_idf,
        len(overlap), bm25_score
    ]

feature_names = [
    'num_q_terms', 'num_q_unique', 'num_d_terms', 'num_d_unique',
    'min_q_idf', 'max_q_idf', 'sum_q_idf',
    'min_d_idf', 'max_d_idf', 'sum_d_idf',
    'overlap', 'bm25_score'
]


model = lgb.Booster(model_file="lambdamart_model.txt")

# Rank Top-K Candidates
output_rows = []
top_k = 20

print("Scoring test queries against all training queries...")
for test_idx, test_row in tqdm(test_df.iterrows(), total=len(test_df)):
    query = test_row["query"]
    query_label = test_row["query_label"]

    features = []
    for doc in train_df["candidate"]:
        features.append(extract_features(query, doc, idf_dict))

    scores = model.predict(np.array(features))
    top_indices = np.argsort(scores)[::-1][:top_k]

    for rank, i in enumerate(top_indices):
        output_rows.append({
            "test_query_id": test_idx,
            "query": query,
            "query_label": query_label,
            "candidate": train_df.iloc[i]["candidate"],
            "candidate_label": train_df.iloc[i]["candidate_label"],
            "score": scores[i],
            "rank": rank + 1
        })


output_df = pd.DataFrame(output_rows)
output_df.to_csv("test_query_top20_candidates.csv", index=False)
print("Saved test_query_top20_candidates.csv")
