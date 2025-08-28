import os
import pickle
import pandas as pd
from tqdm import tqdm

from nq_utils import extract_features, extract_drmm_w2v_features, build_w2v_model

# --------- Paths ----------
TEST_IN = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/nq_open_top15_bm25_llm_pairs_test.csv.gz"
OUT_DIR = os.path.join(os.path.dirname(TEST_IN), "word12_drmmw2v_features")
os.makedirs(OUT_DIR, exist_ok=True)

# Reuse IDF cache (don’t rebuild)
IDF_CACHE_PKL = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/word12_features/idf_from_terrier.pkl"

# Output file
OUT_CSV = os.path.join(
    OUT_DIR,
    os.path.basename(TEST_IN).replace(".csv.gz", "_word12_drmmw2v.csv.gz")
)

# Feature names
WORD12 = [
    'num_q_terms', 'num_q_unique', 'num_d_terms', 'num_d_unique',
    'min_q_idf', 'max_q_idf', 'sum_q_idf',
    'min_d_idf', 'max_d_idf', 'sum_d_idf',
    'overlap', 'bm25_score'
]
DRMM20 = [f"drmm_hist_bin_{i}" for i in range(20)]
FEATURE_NAMES = WORD12 + DRMM20

def main():
    print(f"Loading test CSV: {TEST_IN}")
    df = pd.read_csv(TEST_IN, compression="gzip")
    needed_cols = {"query", "text"}
    if not needed_cols.issubset(df.columns):
        missing = needed_cols - set(df.columns)
        raise ValueError(f"Missing columns in input CSV: {missing}")

    # ---- Load IDF dict ----
    if not os.path.isfile(IDF_CACHE_PKL):
        raise FileNotFoundError(f"Expected IDF cache not found: {IDF_CACHE_PKL}")
    with open(IDF_CACHE_PKL, "rb") as f:
        idf_dict = pickle.load(f)
    print(f"Loaded IDF dict (size={len(idf_dict):,})")

    # ---- Train Word2Vec on test texts (for DRMM-W2V) ----
    print("Training Word2Vec model on test corpus...")
    w2v_corpus = df["text"].fillna("").tolist()
    w2v_model = build_w2v_model(w2v_corpus, vector_size=300, window=5, min_count=2, sg=1, workers=4)

    # ---- Compute 32 features ----
    print("Extracting 12 word + 20 DRMM-w2v features ...")
    feats_list = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        q, d = row["query"], row["text"]
        word12_feats = extract_features(q, d, idf_dict)
        drmm20_feats = extract_drmm_w2v_features(q, d, w2v_model)
        feats_list.append(word12_feats + drmm20_feats)

    feats_df = pd.DataFrame(feats_list, columns=FEATURE_NAMES)
    out_df = pd.concat([df.reset_index(drop=True), feats_df], axis=1)

    # ---- Save ----
    out_df.to_csv(OUT_CSV, index=False, compression="gzip")
    print(f"Saved features to: {OUT_CSV}")
    print("Done.")

if __name__ == "__main__":
    main()

