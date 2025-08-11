import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KDTree
from tqdm import tqdm
# from utils import (
#     extract_features,          # 12 handcrafted
#     extract_drmm_features,     # 20-bin DRMM histogram (uses LDA internally)
#     get_idf_dict,
#     build_lda_model,
# )

# Load datasets
train_df = pd.read_csv("/kaggle/input/top-50-dataset/train.tsv",
                       sep="\t", names=["candidate", "candidate_label"])
test_df  = pd.read_csv("/kaggle/input/top-50-dataset/test.tsv",
                       sep="\t", names=["query", "query_label"])

# 1) SBERT embeddings
print("Generating SBERT embeddings...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
train_embeddings = sbert.encode(train_df["candidate"].tolist(), show_progress_bar=True)
test_embeddings  = sbert.encode(test_df["query"].tolist(), show_progress_bar=True)

# 2) KDTree on training embeddings
tree = KDTree(train_embeddings)

# 3) Build IDF from full training candidates (for handcrafted features)
idf_dict = get_idf_dict(train_df["candidate"].tolist())

# 4) Train LDA model on full training candidates (DRMM needs phi vectors)
print("Training LDA model (for DRMM features only)...")
lda_model, lda_dict = build_lda_model(train_df["candidate"].tolist())

# 5) Fetch top-50 candidates + extract 32 features
top_k = 50
results = []

print("Fetching top 50 SBERT neighbors and extracting 32 features (12 + 20 DRMM)...")
for i, test_row in tqdm(test_df.iterrows(), total=len(test_df)):
    query = test_row["query"]
    query_label = test_row["query_label"]
    q_emb = test_embeddings[i].reshape(1, -1)

    dists, indices = tree.query(q_emb, k=top_k)
    for rank, idx in enumerate(indices[0]):
        cand_row = train_df.iloc[idx]
        candidate = cand_row["candidate"]
        candidate_label = cand_row["candidate_label"]
        distance = float(dists[0][rank])

        # 12 handcrafted + 20 DRMM (no standalone LDA features)
        feats_hand = extract_features(query, candidate, idf_dict)
        feats_drmm = extract_drmm_features(query, candidate, lda_model, lda_dict)
        all_features = feats_hand + feats_drmm  # 32 total

        results.append({
            "test_query_id": i,
            "query": query,
            "query_label": int(query_label),
            "candidate": candidate,
            "candidate_label": int(candidate_label),
            "distance": distance,
            "rank": rank + 1,
            **{f"f_{j+1}": val for j, val in enumerate(all_features)}
        })

# 6) Save to CSV
output_df = pd.DataFrame(results)
output_df.to_csv("test_query_top50_word_DRMM_features.csv", index=False)
print("Saved: test_query_top50_word_DRMM_features.csv")
