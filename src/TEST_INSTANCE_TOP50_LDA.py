import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KDTree
from tqdm import tqdm
#from utils import extract_features, extract_lda_features, get_idf_dict, build_lda_model

# Load datasets
train_df = pd.read_csv("/kaggle/input/top-50-dataset/train.tsv", sep="\t", names=["candidate", "candidate_label"])
test_df = pd.read_csv("/kaggle/input/top-50-dataset/test.tsv", sep="\t", names=["query", "query_label"])

# Generate SBERT embeddings
print("Generating SBERT embeddings...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
train_embeddings = sbert.encode(train_df["candidate"].tolist(), show_progress_bar=True)
test_embeddings = sbert.encode(test_df["query"].tolist(), show_progress_bar=True)

# Build KDTree
tree = KDTree(train_embeddings)

# Build IDF dictionary
idf_dict = get_idf_dict(train_df["candidate"].tolist())

# Train LDA model on combined corpus
print("Training LDA model...")
all_texts = train_df["candidate"].tolist() + test_df["query"].tolist()
lda_model, lda_dict = build_lda_model(all_texts)

# Define feature names
handcrafted_cols = [
    'num_q_terms', 'num_q_unique', 'num_d_terms', 'num_d_unique',
    'min_q_idf', 'max_q_idf', 'sum_q_idf',
    'min_d_idf', 'max_d_idf', 'sum_d_idf',
    'overlap', 'bm25_score'
]
lda_cols = [
    'q_avg_phi', 'q_argmin_norm', 'q_argmax_norm',
    'q_phi_min_sim', 'q_phi_max_sim', 'q_phi_avg_sim',
    'd_avg_phi', 'd_argmin_norm', 'd_argmax_norm',
    'd_phi_min_sim', 'd_phi_max_sim', 'd_phi_avg_sim',
    'qd_single_link', 'qd_complete_link', 'qd_avg_link',
    'theta_cos_sim'
]
feature_cols = handcrafted_cols + lda_cols

# Fetch top-50 candidates per query
top_k = 50
results = []

print("Fetching top 50 SBERT neighbors and extracting features...")
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

        # Extract features
        handcrafted_feats = extract_features(query, candidate, idf_dict)
        lda_feats = extract_lda_features(query, candidate, lda_model, lda_dict)
        full_feats = handcrafted_feats + lda_feats

        # Append row
        results.append({
            "test_query_id": i,
            "query": query,
            "query_label": query_label,
            "candidate": candidate,
            "candidate_label": candidate_label,
            "distance": distance,
            "rank": rank + 1,
            **{col: val for col, val in zip(feature_cols, full_feats)}
        })

# Save output
output_df = pd.DataFrame(results)
output_df.to_csv("test_query_top50_candidates_with_16_LDA_features.csv", index=False)
print("Saved: test_query_top50_candidates_with_16_LDA_features.csv")
