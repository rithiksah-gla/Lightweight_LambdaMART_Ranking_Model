import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KDTree
from tqdm import tqdm
from utils import extract_features, get_idf_dict

# Load datasets
train_df = pd.read_csv("/kaggle/input/top-50-dataset/train.tsv", sep="\t", names=["candidate", "candidate_label"])
test_df = pd.read_csv("/kaggle/input/top-50-dataset/test.tsv", sep="\t", names=["query", "query_label"])

# SBERT Embeddings
print("Generating SBERT embeddings...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
train_embeddings = sbert.encode(train_df["candidate"].tolist(), show_progress_bar=True)
test_embeddings = sbert.encode(test_df["query"].tolist(), show_progress_bar=True)

# KDTree on training data
tree = KDTree(train_embeddings)

# Compute IDF from training texts
idf_dict = get_idf_dict(train_df["candidate"].tolist())

# Fetch top-50 candidates for each test query
top_k = 50
results = []

print("Fetching top 50 SBERT neighbors...")
for i, test_row in tqdm(test_df.iterrows(), total=len(test_df)):
    query = test_row["query"]
    query_label = test_row["query_label"]
    query_embedding = test_embeddings[i].reshape(1, -1)

    dists, indices = tree.query(query_embedding, k=top_k)
    for rank, idx in enumerate(indices[0]):
        candidate_row = train_df.iloc[idx]
        candidate = candidate_row["candidate"]
        candidate_label = candidate_row["candidate_label"]
        distance = dists[0][rank]

        feats = extract_features(query, candidate, idf_dict)

        results.append({
            "test_query_id": i,
            "query": query,
            "query_label": query_label,
            "candidate": candidate,
            "candidate_label": candidate_label,
            "distance": distance,
            "rank": rank + 1,
            **{name: val for name, val in zip([
                'num_q_terms', 'num_q_unique', 'num_d_terms', 'num_d_unique',
                'min_q_idf', 'max_q_idf', 'sum_q_idf',
                'min_d_idf', 'max_d_idf', 'sum_d_idf',
                'overlap', 'bm25_score'
            ], feats)}
        })

# Save to CSV
output_df = pd.DataFrame(results)
output_df.to_csv("test_query_top50_12_features.csv", index=False)
print("Saved: test_query_top50_12_features_candidates.csv")
