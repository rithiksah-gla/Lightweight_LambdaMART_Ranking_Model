import os
import pickle
import pandas as pd
from tqdm import tqdm

from nq_utils import (
    extract_features,
    build_lda_model, extract_lda_features,
    extract_drmm_features
)

# -------- Paths --------
TEST_IN = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/nq_open_top15_bm25_llm_pairs_test.csv.gz"
OUT_DIR = os.path.join(os.path.dirname(TEST_IN), "word_lda_drmm_features")
os.makedirs(OUT_DIR, exist_ok=True)

IDF_CACHE_PKL = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/word12_features/idf_from_terrier.pkl"

OUT_CSV = os.path.join(
    OUT_DIR,
    os.path.basename(TEST_IN).replace(".csv.gz", "_word_lda_drmm.csv.gz")
)

WORD12 = [
    'num_q_terms','num_q_unique','num_d_terms','num_d_unique',
    'min_q_idf','max_q_idf','sum_q_idf',
    'min_d_idf','max_d_idf','sum_d_idf',
    'overlap','bm25_score'
]
LDA16 = [f"lda_{i+1}" for i in range(16)]
DRMM20 = [f"drmm_lda_bin_{i}" for i in range(20)]
FEATURE_NAMES = WORD12 + LDA16 + DRMM20

def main():
    print(f"Loading test CSV: {TEST_IN}")
    df = pd.read_csv(TEST_IN, compression="gzip")
    needed = {"query","text"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing columns: {needed - set(df.columns)}")

    # IDF
    if not os.path.isfile(IDF_CACHE_PKL):
        raise FileNotFoundError(f"IDF cache not found: {IDF_CACHE_PKL}")
    with open(IDF_CACHE_PKL, "rb") as f:
        idf_dict = pickle.load(f)
    print(f"Loaded IDF ({len(idf_dict):,})")

    # LDA
    print("Training LDA on test corpus (texts + queries)...")
    lda_corpus = pd.concat([df["text"].fillna(""), df["query"].fillna("")], axis=0).tolist()
    lda_model, lda_dict = build_lda_model(lda_corpus, num_topics=50)

    # Compute features
    print("Extracting 48 features (word12 + lda16 + drmm20)...")
    feats = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        q, d = str(r["query"]), str(r["text"])
        f_word = extract_features(q, d, idf_dict)
        f_lda  = extract_lda_features(q, d, lda_model, lda_dict, num_topics=50)
        f_drmm = extract_drmm_features(q, d, lda_model, lda_dict, num_topics=50, num_bins=20)
        feats.append(f_word + f_lda + f_drmm)

    feats_df = pd.DataFrame(feats, columns=FEATURE_NAMES)
    out_df = pd.concat([df.reset_index(drop=True), feats_df], axis=1)
    out_df.to_csv(OUT_CSV, index=False, compression="gzip")
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()
