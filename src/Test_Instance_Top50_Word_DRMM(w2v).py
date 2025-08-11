# precompute_top50_32_w2v.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KDTree
from tqdm import tqdm
#from utils import extract_features, get_idf_dict, build_w2v_model, extract_drmm_w2v_features

# Load datasets
train_df = pd.read_csv("/kaggle/input/top-50-dataset/train.tsv", sep="\t", names=["candidate", "candidate_label"])
test_df  = pd.read_csv("/kaggle/input/top-50-dataset/test.tsv",  sep="\t", names=["query", "query_label"])

# SBERT embeddings
print("Generating SBERT embeddings...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
train_embeddings = sbert.encode(train_df["candidate"].tolist(), show_progress_bar=True)
test_embeddings  = sbert.encode(test_df["query"].tolist(), show_progress_bar=True)

# KDTree on training candidates
tree = KDTree(train_embeddings)

# Build IDF on train.tsv (candidates)
idf_dict = get_idf_dict(train_df["candidate"].tolist())

# Train Word2Vec on train.tsv (candidates)
print("Training Word2Vec model...")
w2v_model = build_w2v_model(train_df["candidate"].tolist())

# Fetch top-50 + extract 32 features
top_k = 50
results = []
print("Fetching top-50 and extracting 32 (12 + DRMM-W2V) features...")
for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
    query = row["query"]; q_label = row["query_label"]
    q_emb = test_embeddings[i].reshape(1, -1)
    dists, idxs = tree.query(q_emb, k=top_k)

    for rank, idx in enumerate(idxs[0]):
        cand_row = train_df.iloc[idx]
        cand, c_label = cand_row["candidate"], cand_row["candidate_label"]
        distance = float(dists[0][rank])

        feats_12 = extract_features(query, cand, idf_dict)
        feats_drmm = extract_drmm_w2v_features(query, cand, w2v_model)  # 20 dims

        all_feats = feats_12 + feats_drmm  # 32 dims

        out = {
            "test_query_id": i,
            "query": query,
            "query_label": int(q_label),
            "candidate": cand,
            "candidate_label": int(c_label),
            "distance": distance,
            "rank": rank + 1,
        }
        # write features as f_1..f_32
        for j, v in enumerate(all_feats, start=1):
            out[f"f_{j}"] = v
        results.append(out)

out_df = pd.DataFrame(results)
out_df.to_csv("test_query_top50_word_DRMM_w2v_features.csv", index=False)
print("Saved: test_query_top50_word_DRMM_w2v_features.csv")
