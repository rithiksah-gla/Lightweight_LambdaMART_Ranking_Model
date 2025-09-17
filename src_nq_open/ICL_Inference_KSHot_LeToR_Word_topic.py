import os, pickle
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
import torch

from nq_utils import (
    tokenizer, model, device, compute_f1, sanitize_query,
    extract_features, build_lda_model, extract_lda_features
)

# --------------------------
# Config
# --------------------------
TEST_IN = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/nq_open_top15_bm25_llm_pairs_test.csv.gz"
FEATURED_TEST_IN = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/word_lda28_features/nq_open_top15_bm25_llm_pairs_test_word_lda28.csv.gz"
IDF_CACHE_PKL = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/word12_features/idf_from_terrier.pkl"
MODEL_FILE_DESC = "lambdamart_nq_word_lda_desc.txt"  # use DESC for both orders if you like

OUT_DIR = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/icl_eval_word_lda28"
os.makedirs(OUT_DIR, exist_ok=True)

TOP_K = 10

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

PROMPT = (
    "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) "
    "from one of the provided documents. If none of the documents contain the answer, respond with NO-$"
)

def build_prompt(query, docs):
    s = f"{PROMPT}\nDocuments:\n"
    for d in docs:
        s += f"Document: {d}\n"
    s += f"Question: {query}\nAnswer:"
    return s

def llm_generate_answer(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1900).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, num_beams=1)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("Answer:")[-1].strip()

def ensure_features(df, idf_dict, lda_model, lda_dict):
    if all(c in df.columns for c in FEATURE_NAMES):
        return df
    feats = df.apply(
        lambda r: extract_features(str(r["query"]), str(r["text"]), idf_dict) +
                  extract_lda_features(str(r["query"]), str(r["text"]), lda_model, lda_dict, 50),
        axis=1, result_type="expand"
    )
    feats.columns = FEATURE_NAMES
    return pd.concat([df.reset_index(drop=True), feats], axis=1)

def run(order_label, model_ltr, df, idf_dict, lda_model, lda_dict):
    rows = []
    for qid, group in tqdm(df.groupby("qid"), desc=f"ICL Inference ({order_label.upper()})"):
        query = sanitize_query(str(group.iloc[0]["query"]))
        gold  = group.iloc[0]["ground_truth"]
        try:
            gold = eval(gold) if isinstance(gold, str) else gold
        except Exception:
            gold = [str(gold)]

        g = ensure_features(group, idf_dict, lda_model, lda_dict).copy()
        g["lgbm_score"] = model_ltr.predict(g[FEATURE_NAMES].to_numpy())
        topk = g.sort_values("lgbm_score", ascending=(order_label=="asc")).head(TOP_K)

        docs = topk["text"].fillna("").tolist()
        pred = llm_generate_answer(build_prompt(query, docs))
        prf  = compute_f1(pred, gold)

        rows.append({
            "qid": qid, "query": query, "gold": gold,
            "pred": pred,
            "precision": prf["precision"],
            "recall":    prf["recall"],
            "f1":        prf["f1"]
        })

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, f"icl_infer_word_lda28_{order_label}.csv.gz")
    out_df.to_csv(out_csv, index=False, compression="gzip")

    P, R = out_df["precision"].mean(), out_df["recall"].mean()
    F1 = 0.0 if (P+R)==0 else 2*P*R/(P+R)
    print(f"\nFinal Results ({order_label.upper()}): P={P:.4f} R={R:.4f} F1={F1:.4f}")
    print("Saved:", out_csv)

def main():
    # load test
    if os.path.exists(FEATURED_TEST_IN):
        df = pd.read_csv(FEATURED_TEST_IN, compression="gzip")
    else:
        df = pd.read_csv(TEST_IN, compression="gzip")

    with open(IDF_CACHE_PKL, "rb") as f:
        idf_dict = pickle.load(f)

    # LDA model for this inference run
    if all(c in df.columns for c in FEATURE_NAMES):
        lda_model = lda_dict = None  # not needed
    else:
        print("Training LDA model on test corpus (features not found in CSV)...")
        lda_model, lda_dict = build_lda_model(df["text"].fillna("").tolist(), num_topics=50)

    model_ltr = lgb.Booster(model_file=MODEL_FILE_DESC)

    run("desc", model_ltr, df, idf_dict, lda_model, lda_dict)
    run("asc",  model_ltr, df, idf_dict, lda_model, lda_dict)

if __name__ == "__main__":
    main()
