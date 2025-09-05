import os
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

from nq_utils import (
    extract_features,                 # 12 word
    extract_drmm_w2v_features,        # 20 DRMM-W2V
    extract_lda_features,             # 16 LDA topic features
    build_w2v_model,
    build_lda_model
)

# ---- Paths ----
train_path = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/nq_open_top15_bm25_llm_pairs_train.csv.gz"
val_path   = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/nq_open_top15_bm25_llm_pairs_validation.csv.gz"

# Reuse the IDF cache you already built
IDF_CACHE_PKL = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/word12_features/idf_from_terrier.pkl"

# Feature model caches (so train & inference use same W2V/LDA)
CACHE_DIR = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/feature_caches"
os.makedirs(CACHE_DIR, exist_ok=True)
W2V_MODEL_PATH = os.path.join(CACHE_DIR, "w2v_model.kv")
LDA_MODEL_PATH = os.path.join(CACHE_DIR, "lda_model.gensim")
LDA_DICT_PATH  = os.path.join(CACHE_DIR, "lda_dict.gensim")

# ---- Load data ----
print("Loading train/val CSVs...")
train_df = pd.read_csv(train_path, compression="gzip")[["query", "text", "llm_f1"]]
val_df   = pd.read_csv(val_path,   compression="gzip")[["query", "text", "llm_f1"]]

# ---- Load IDF dict ----
with open(IDF_CACHE_PKL, "rb") as f:
    idf_dict = pickle.load(f)
print(f"Loaded IDF (|V|={len(idf_dict):,})")

# ---- Build / Load W2V ----
if os.path.exists(W2V_MODEL_PATH):
    from gensim.models import KeyedVectors
    w2v_model = KeyedVectors.load(W2V_MODEL_PATH, mmap='r')
    print("Loaded W2V model from cache")
else:
    print("Training W2V on train+val texts...")
    w2v_corpus = pd.concat([train_df["text"], val_df["text"]], axis=0).fillna("").tolist()
    w2v_model = build_w2v_model(w2v_corpus, vector_size=300, window=5, min_count=2, sg=1, workers=4)
    w2v_model.wv.save(W2V_MODEL_PATH)
    print("Saved W2V to:", W2V_MODEL_PATH)

# ---- Build / Load LDA ----
if os.path.exists(LDA_MODEL_PATH) and os.path.exists(LDA_DICT_PATH):
    from gensim import models, corpora
    lda_model = models.LdaModel.load(LDA_MODEL_PATH)
    lda_dict  = corpora.Dictionary.load(LDA_DICT_PATH)
    print("Loaded LDA model & dict from cache")
else:
    print("Training LDA on train+val texts...")
    lda_corpus = pd.concat([train_df["text"], val_df["text"]], axis=0).fillna("").tolist()
    lda_model, lda_dict = build_lda_model(lda_corpus, num_topics=50)
    lda_model.save(LDA_MODEL_PATH)
    lda_dict.save(LDA_DICT_PATH)
    print("Saved LDA model:", LDA_MODEL_PATH)
    print("Saved LDA dict:",  LDA_DICT_PATH)

# ---- Feature names ----
WORD12 = [
    'num_q_terms','num_q_unique','num_d_terms','num_d_unique',
    'min_q_idf','max_q_idf','sum_q_idf',
    'min_d_idf','max_d_idf','sum_d_idf',
    'overlap','bm25_score'
]
DRMM20 = [f"drmm_hist_bin_{i}" for i in range(20)]
LDA16  = [f"lda_feat_{i+1}" for i in range(16)]
FEATURE_NAMES = WORD12 + DRMM20 + LDA16

def add_features(df):
    feats = df.apply(
        lambda r: extract_features(r["query"], r["text"], idf_dict)
                  + extract_drmm_w2v_features(r["query"], r["text"], w2v_model)
                  + extract_lda_features(r["query"], r["text"], lda_model, lda_dict, num_topics=50),
        axis=1, result_type='expand'
    )
    feats.columns = FEATURE_NAMES
    return pd.concat([df, feats], axis=1)

print("Extracting 48 features (12+20+16)...")
train_df = add_features(train_df)
val_df   = add_features(val_df)

def assign_relevance_from_llm_f1(df, order="desc"):
    ascending = (order == "asc")
    df = df.sort_values(by=["query", "llm_f1"], ascending=[True, ascending]).copy()
    df["rwi"] = df.groupby("query").cumcount()
    max_per_q = df.groupby("query")["rwi"].transform("max")
    df["relevance"] = max_per_q - df["rwi"]
    return df.drop(columns=["rwi"])

train_desc = assign_relevance_from_llm_f1(train_df, order="desc")
val_desc   = assign_relevance_from_llm_f1(val_df,   order="desc")
train_asc  = assign_relevance_from_llm_f1(train_df, order="asc")
val_asc    = assign_relevance_from_llm_f1(val_df,   order="asc")

def train_and_save_model(train_df, val_df, model_name):
    X_train = train_df[FEATURE_NAMES]; y_train = train_df["relevance"]
    X_val   = val_df[FEATURE_NAMES];   y_val   = val_df["relevance"]
    gtr = train_df.groupby("query").size().to_list()
    gva = val_df.groupby("query").size().to_list()

    dtr = lgb.Dataset(X_train, label=y_train, group=gtr)
    dva = lgb.Dataset(X_val,   label=y_val,   group=gva, reference=dtr)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1,3,5,10,15],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_pre_filter": False,
        "verbosity": -1,
        "label_gain": list(range(int(max(y_train.max(), y_val.max())) + 1))
    }
    print(f"\nTraining {model_name} ...")
    model = lgb.train(
        params, dtr, num_boost_round=2000,
        valid_sets=[dtr, dva], valid_names=["train","val"],
        callbacks=[early_stopping(50), log_evaluation(50)]
    )
    model.save_model(model_name)
    print("Saved:", model_name)

train_and_save_model(train_desc, val_desc, "lambdamart_nq_word12_drmmw2v_lda16_desc.txt")
train_and_save_model(train_asc,  val_asc,  "lambdamart_nq_word12_drmmw2v_lda16_asc.txt")
print("Done.")
