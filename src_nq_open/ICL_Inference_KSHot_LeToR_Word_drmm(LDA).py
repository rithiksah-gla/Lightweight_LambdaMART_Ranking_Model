# icl_infer_nq_word12_lda_drmm.py
import os, json, pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
import torch

from nq_utils import (
    tokenizer, model, device,
    compute_f1, sanitize_query,
    extract_features, build_lda_model, extract_drmm_features
)

# --------------------------
# Config
# --------------------------
TEST_IN = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/nq_open_top15_bm25_llm_pairs_test.csv.gz"
FEATURED_TEST_IN = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/word12_lda_drmm_features/nq_open_top15_bm25_llm_pairs_test_word12_lda_drmm.csv.gz"
IDF_CACHE_PKL = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/word12_features/idf_from_terrier.pkl"
MODEL_FILE_DESC = "lambdamart_nq_word_lda_drmm_desc.txt"

OUT_DIR = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/icl_eval_word12_lda_drmm"
os.makedirs(OUT_DIR, exist_ok=True)

TOP_K = 10   # demonstration shots
WORD12 = [
    'num_q_terms','num_q_unique','num_d_terms','num_d_unique',
    'min_q_idf','max_q_idf','sum_q_idf',
    'min_d_idf','max_d_idf','sum_d_idf',
    'overlap','bm25_score'
]
DRMM20 = [f"drmm_hist_bin_{i}" for i in range(20)]
FEATURE_NAMES = WORD12 + DRMM20  # 32

PROMPT_INSTRUCTION = (
    "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) "
    "from one of the provided documents. If none of the documents contain the answer, respond with NO-$"
)

# --------------------------
# Helpers
# --------------------------
def build_prompt(query, docs):
    prompt = f"{PROMPT_INSTRUCTION}\nDocuments:\n"
    for d in docs:
        prompt += f"Document: {d}\n"
    prompt += f"Question: {query}\nAnswer:"
    return prompt

def llm_generate_answer(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1900).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, num_beams=1)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("Answer:")[-1].strip()

def ensure_features(df, idf_dict, lda_model, lda_dict):
    if all(c in df.columns for c in FEATURE_NAMES):
        return df
    rows = []
    for _, r in df.iterrows():
        q, d = str(r["query"]), str(r["text"])
        rows.append(
            extract_features(q, d, idf_dict) +
            extract_drmm_features(q, d, lda_model, lda_dict, num_topics=50, num_bins=20)
        )
    feats = pd.DataFrame(rows, columns=FEATURE_NAMES)
    return pd.concat([df.reset_index(drop=True), feats], axis=1)

def run_inference(order_label, model_ltr, df, idf_dict, lda_model, lda_dict):
    results = []
    for qid, group in tqdm(df.groupby("qid"), desc=f"ICL Inference ({order_label.upper()})"):
        query = sanitize_query(str(group.iloc[0]["query"]))
        gold = group.iloc[0]["ground_truth"]
        try:
            gold = eval(gold) if isinstance(gold, str) else gold
        except Exception:
            gold = [str(gold)]

        g = ensure_features(group, idf_dict, lda_model, lda_dict).copy()
        g["lgbm_score"] = model_ltr.predict(g[FEATURE_NAMES].to_numpy())

        if order_label == "desc":
            topk = g.sort_values("lgbm_score", ascending=False).head(TOP_K)
        else:  # asc
            topk = g.sort_values("lgbm_score", ascending=True).head(TOP_K)

        docs = topk["text"].fillna("").tolist()
        prompt = build_prompt(query, docs)
        pred = llm_generate_answer(prompt)

        prf = compute_f1(pred, gold)
        results.append({
            "qid": qid, "query": query, "gold": gold,
            "pred": pred,
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"]
        })

    out_df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR, f"icl_infer_word12_lda_drmm_{order_label}.csv.gz")
    out_df.to_csv(out_csv, index=False, compression="gzip")

    P, R = out_df["precision"].mean(), out_df["recall"].mean()
    F1 = 0.0 if (P+R)==0 else 2*P*R/(P+R)
    print(f"\nFinal Results ({order_label.upper()}): P={P:.4f} R={R:.4f} F1={F1:.4f}")
    print("Saved:", out_csv)

# --------------------------
# Main
# --------------------------
def main():
    # load test set
    if os.path.exists(FEATURED_TEST_IN):
        df = pd.read_csv(FEATURED_TEST_IN, compression="gzip")
    else:
        df = pd.read_csv(TEST_IN, compression="gzip")

    # idf dict
    with open(IDF_CACHE_PKL, "rb") as f:
        idf_dict = pickle.load(f)

    need_lda = not all(c in df.columns for c in FEATURE_NAMES)
    if need_lda:
        print("Training LDA on test texts (features missing in CSV)...")
        lda_model, lda_dict = build_lda_model(df["text"].fillna("").tolist(), num_topics=50)
    else:
        # dummy minimal to satisfy signature; will never be used
        lda_model = None; lda_dict = None

    # model
    model_ltr = lgb.Booster(model_file=MODEL_FILE_DESC)

    # run both DESC and ASC (DESC model for both)
    run_inference("desc", model_ltr, df, idf_dict, lda_model, lda_dict)
    run_inference("asc",  model_ltr, df, idf_dict, lda_model, lda_dict)

if __name__ == "__main__":
    main()
