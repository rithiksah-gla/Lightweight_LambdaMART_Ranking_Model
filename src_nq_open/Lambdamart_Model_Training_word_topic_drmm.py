import os
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

from nq_utils import (
    extract_features,          # 12 word
    build_lda_model,           # LDA trainer
    extract_lda_features,      # 16 LDA
    extract_drmm_features      # 20 DRMM from LDA
)

# -------- Paths --------
train_path = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/nq_open_top15_bm25_llm_pairs_train.csv.gz"
val_path   = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/nq_open_top15_bm25_llm_pairs_validation.csv.gz"

IDF_CACHE_PKL = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/word12_features/idf_from_terrier.pkl"

# -------- Load data --------
print("Loading train/val CSVs...")
train_df = pd.read_csv(train_path, compression="gzip")[["query", "text", "llm_f1"]]
val_df   = pd.read_csv(val_path,   compression="gzip")[["query", "text", "llm_f1"]]

# -------- Load IDF --------
with open(IDF_CACHE_PKL, "rb") as f:
    idf_dict = pickle.load(f)
print(f"Loaded IDF dict (size={len(idf_dict):,})")

# -------- Train LDA --------
print("Training LDA on training corpus (texts + queries)...")
lda_corpus = pd.concat(
    [train_df["text"].fillna(""), train_df["query"].fillna("")],
    axis=0
).tolist()
lda_model, lda_dict = build_lda_model(lda_corpus, num_topics=50)
print("LDA ready.")

WORD12 = [
    'num_q_terms','num_q_unique','num_d_terms','num_d_unique',
    'min_q_idf','max_q_idf','sum_q_idf',
    'min_d_idf','max_d_idf','sum_d_idf',
    'overlap','bm25_score'
]
LDA16 = [f"lda_{i+1}" for i in range(16)]
DRMM20 = [f"drmm_lda_bin_{i}" for i in range(20)]
FEATURE_NAMES = WORD12 + LDA16 + DRMM20

def add_features(df):
    feats = []
    for _, r in df.iterrows():
        q, d = str(r["query"]), str(r["text"])
        f_word = extract_features(q, d, idf_dict)
        f_lda  = extract_lda_features(q, d, lda_model, lda_dict, num_topics=50)
        f_drmm = extract_drmm_features(q, d, lda_model, lda_dict, num_topics=50, num_bins=20)
        feats.append(f_word + f_lda + f_drmm)
    feat_df = pd.DataFrame(feats, columns=FEATURE_NAMES)
    return pd.concat([df.reset_index(drop=True), feat_df], axis=1)

print("Extracting 48 features for train/val...")
train_df = add_features(train_df)
val_df   = add_features(val_df)

def assign_relevance_from_llm_f1(df, order="desc"):
    ascending = (order == "asc")
    df = df.sort_values(by=["query","llm_f1"], ascending=[True, ascending]).copy()
    df["rank_within"] = df.groupby("query").cumcount()
    max_per_q = df.groupby("query")["rank_within"].transform("max")
    df["relevance"] = max_per_q - df["rank_within"]
    return df.drop(columns=["rank_within"])

train_desc = assign_relevance_from_llm_f1(train_df, order="desc")
val_desc   = assign_relevance_from_llm_f1(val_df,   order="desc")
train_asc  = assign_relevance_from_llm_f1(train_df, order="asc")
val_asc    = assign_relevance_from_llm_f1(val_df,   order="asc")

def train_and_save(train_df, val_df, model_name):
    X_tr, y_tr = train_df[FEATURE_NAMES], train_df["relevance"]
    X_va, y_va = val_df[FEATURE_NAMES],   val_df["relevance"]
    g_tr = train_df.groupby("query").size().tolist()
    g_va = val_df.groupby("query").size().tolist()

    dtr = lgb.Dataset(X_tr, label=y_tr, group=g_tr)
    dva = lgb.Dataset(X_va, label=y_va, group=g_va, reference=dtr)

    params = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1,3,5,10,15],
        "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 20,
        "feature_pre_filter": False, "verbosity": -1,
        "label_gain": list(range(int(max(y_tr.max(), y_va.max())) + 1))
    }

    print(f"\nTraining {model_name} ...")
    model = lgb.train(
        params, dtr, num_boost_round=2000,
        valid_sets=[dtr, dva], valid_names=["train","val"],
        callbacks=[early_stopping(50), log_evaluation(50)]
    )
    model.save_model(model_name)
    print("Saved:", model_name)

train_and_save(train_desc, val_desc, "lambdamart_nq_word_lda_drmm_desc.txt")
train_and_save(train_asc,  val_asc,  "lambdamart_nq_word_lda_drmm_asc.txt")
print("Done.")
