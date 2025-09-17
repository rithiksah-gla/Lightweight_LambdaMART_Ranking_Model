import os
import pickle
import pandas as pd
from tqdm import tqdm

from nq_utils import extract_features, build_lda_model, extract_lda_features

# --------- Paths ----------
TEST_IN = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/nq_open_top15_bm25_llm_pairs_test.csv.gz"

OUT_DIR = os.path.join(os.path.dirname(TEST_IN), "word_lda28_features")
os.makedirs(OUT_DIR, exist_ok=True)

# Reuse IDF cache (built earlier)
IDF_CACHE_PKL = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/word12_features/idf_from_terrier.pkl"

# Output file
OUT_CSV = os.path.join(
    OUT_DIR,
    os.path.basename(TEST_IN).replace(".csv.gz", "_word_lda28.csv.gz")
)

# Feature names
WORD12 = [
    'num_q_terms','num_q_unique','num_d_terms','num_d_unique',
    'min_q_idf','max_q_idf','sum_q_idf',
    'min_d_idf','max_d_idf','sum_d_idf',
    'overlap','bm25_score'
]
LDA16 = [
    'lda_qavg_norm','lda_q_min_norm','lda_q_max_norm','lda_q_min_sim','lda_q_max_sim','lda_q_avg_sim',
    'lda_davg_norm','lda_d_min_norm','lda_d_max_norm','lda_d_min_sim','lda_d_max_sim','lda_d_avg_sim',
    'lda_sl_sim','lda_cl_sim','lda_al_sim','lda_theta_sim'
]
FEATURE_NAMES = WORD12 + LDA16

def main():
    print(f"Loading test CSV: {TEST_IN}")
    df = pd.read_csv(TEST_IN, compression="gzip")
    needed = {"query", "text"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing columns: {needed - set(df.columns)}")

    # Load IDF dict
    if not os.path.isfile(IDF_CACHE_PKL):
        raise FileNotFoundError(f"Expected IDF cache not found: {IDF_CACHE_PKL}")
    with open(IDF_CACHE_PKL, "rb") as f:
        idf_dict = pickle.load(f)
    print(f"Loaded IDF dict (size={len(idf_dict):,})")

    # Train LDA on test texts
    print("Training LDA model on test corpus...")
    lda_model, lda_dict = build_lda_model(df["text"].fillna("").tolist(), num_topics=50)

    print("Extracting 28 (Word+LDA) features ...")
    feats = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        q, d = r["query"], r["text"]
        feats.append(
            extract_features(q, d, idf_dict) +
            extract_lda_features(q, d, lda_model, lda_dict, 50)
        )

    feats_df = pd.DataFrame(feats, columns=FEATURE_NAMES)
    out_df = pd.concat([df.reset_index(drop=True), feats_df], axis=1)
    out_df.to_csv(OUT_CSV, index=False, compression="gzip")
    print(f"Saved features to: {OUT_CSV}")

if __name__ == "__main__":
    main()
