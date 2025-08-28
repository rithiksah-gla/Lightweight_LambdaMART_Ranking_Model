import os
import pickle
import pandas as pd
from tqdm import tqdm
from nq_utils import extract_features, build_idf_from_terrier_index

# Paths
TEST_IN = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/nq_open_top15_bm25_llm_pairs_test.csv.gz"
INDEX_DIR = "/scratch/2980356s/data/SST2/wiki2018_nqopen"  # Terrier index (wiki2018 nq-open)

# We'll write outputs alongside the input file, in a NEW subfolder
OUT_DIR = os.path.join(os.path.dirname(TEST_IN), "word12_features")
os.makedirs(OUT_DIR, exist_ok=True)

# IDF cache (so we don't rebuild every time)
IDF_CACHE_PKL = os.path.join(OUT_DIR, "idf_from_terrier.pkl")

# Output file
OUT_CSV = os.path.join(
    OUT_DIR,
    os.path.basename(TEST_IN).replace(".csv.gz", "_word12.csv.gz")
)

# 12 handcrafted word features (for column naming)
FEATURE_NAMES = [
    'num_q_terms', 'num_q_unique', 'num_d_terms', 'num_d_unique',
    'min_q_idf', 'max_q_idf', 'sum_q_idf',
    'min_d_idf', 'max_d_idf', 'sum_d_idf',
    'overlap', 'bm25_score'
]

def main():
    print(f"Loading test CSV: {TEST_IN}")
    # Expecting columns: query, text, llm_f1, llm_ans, ground_truth
    df = pd.read_csv(TEST_IN, compression="gzip")
    needed_cols = {"query", "text"}
    if not needed_cols.issubset(df.columns):
        missing = needed_cols - set(df.columns)
        raise ValueError(f"Missing columns in input CSV: {missing}")

    # ---- Build / load IDF from Terrier index ----
    if os.path.isfile(IDF_CACHE_PKL):
        print(f"Loading cached IDF from: {IDF_CACHE_PKL}")
        with open(IDF_CACHE_PKL, "rb") as f:
            idf_dict = pickle.load(f)
    else:
        print(f"Building IDF from Terrier index: {INDEX_DIR}")
        # You can pass sample_docs=<int> to speed it up for a quick run
        idf_dict = build_idf_from_terrier_index(
            index_dir=INDEX_DIR,
            meta_field="text",
            sample_docs=None  # set to e.g. 2_000_000 for approx IDF if needed
        )
        print(f"IDF vocab size: {len(idf_dict):,}")
        with open(IDF_CACHE_PKL, "wb") as f:
            pickle.dump(idf_dict, f)
        print(f"Saved IDF cache to: {IDF_CACHE_PKL}")

    # ---- Compute 12 features ----
    print("Extracting 12 word features for each (query, text) row ...")
    # tqdm + apply for a little progress feedback
    feats_list = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        feats = extract_features(row["query"], row["text"], idf_dict)
        feats_list.append(feats)

    feats_df = pd.DataFrame(feats_list, columns=FEATURE_NAMES)
    out_df = pd.concat([df.reset_index(drop=True), feats_df], axis=1)
  
    out_df.to_csv(OUT_CSV, index=False, compression="gzip")
    print(f"Saved features to: {OUT_CSV}")
    print("Done.")

if __name__ == "__main__":
    main()

