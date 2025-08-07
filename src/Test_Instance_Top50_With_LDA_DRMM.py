import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KDTree
from tqdm import tqdm
from utils import extract_features, extract_lda_features, extract_drmm_features, get_idf_dict, build_lda_model, tokenize

# Load datasets
train_df = pd.read_csv("/kaggle/input/top-50-dataset/train.tsv", sep="\t", names=["candidate", "candidate_label"])
test_df = pd.read_csv("/kaggle/input/top-50-dataset/test.tsv", sep="\t", names=["query", "query_label"])

# Step 1: Generate SBERT embeddings
print("Generating SBERT embeddings...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
train_embeddings = sbert.encode(train_df["candidate"].tolist(), show_progress_bar=True)
test_embeddings = sbert.encode(test_df["query"].tolist(), show_progress_bar=True)

# Step 2: KDTree on training embeddings
tree = KDTree(train_embeddings)

# Step 3: Create IDF dictionary from training candidates
idf_dict = get_idf_dict(train_df["candidate"].tolist())

# Step 4: Train LDA model using combined text
print("Training LDA model...")
lda_model, lda_dict = build_lda_model(train_df["candidate"].tolist())

# Step 5: Fetch top-50 candidates for each test query
top_k = 50
results = []

print("Fetching top 50 SBERT neighbors and extracting 48 features...")
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
        handcrafted = extract_features(query, candidate, idf_dict)
        lda = extract_lda_features(query, candidate, lda_model, lda_dict)
        drmm = extract_drmm_features(query, candidate, lda_model, lda_dict)

        # Combine all
        all_features = handcrafted + lda + drmm

        results.append({
            "test_query_id": i,
            "query": query,
            "query_label": query_label,
            "candidate": candidate,
            "candidate_label": candidate_label,
            "distance": distance,
            "rank": rank + 1,
            **{f"f_{j+1}": val for j, val in enumerate(all_features)}
        })

# Save to CSV
output_df = pd.DataFrame(results)
output_df.to_csv("test_query_top50_48_features.csv", index=False)
print("Saved: test_query_top50_48_features.csv")
