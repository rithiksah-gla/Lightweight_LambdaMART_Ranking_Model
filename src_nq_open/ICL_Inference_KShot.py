# baseline_k_shot_from_csv.py
import os
import pandas as pd
import torch
from tqdm import tqdm
from nq_utils_1 import tokenizer, model, device, compute_f1, sanitize_query

TEST_PATH = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/nq_open_top15_bm25_llm_pairs_test.csv.gz"
OUT_DIR   = "./baselines"
os.makedirs(OUT_DIR, exist_ok=True)

K = 10
PROMPT_INSTRUCTION = (
    "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) "
    "from one of the provided documents. If none of the documents contain the answer, respond with NO-$"
)
MAX_NEW_TOKENS = 50

def build_prompt(question: str, docs) -> str:
    # Keep the exact instruction; add multiple Document: lines
    prompt = f"{PROMPT_INSTRUCTION}\nDocuments:\n"
    for t in docs:
        prompt += f"Document: {t}\n"
    prompt += f"Question: {question}\nAnswer:"
    return prompt

def llm_answer(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1900).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, num_beams=1)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    tail = decoded.split("Answer:")[-1] if "Answer:" in decoded else decoded
    return tail.strip()

def run_kshot(order="desc"):
    df = pd.read_csv(TEST_PATH, compression="gzip")
    results = []

    asc = (order == "asc")

    for query, group in tqdm(df.groupby("query"), desc=f"{K}-shot {order.upper()}"):
        q = sanitize_query(str(query))
        gt = group.iloc[0]["ground_truth"]
        try:
            gold = eval(gt) if isinstance(gt, str) else gt
        except Exception:
            gold = [str(gt)]

        # choose K docs by llm_f1
        chosen = group.sort_values("llm_f1", ascending=asc).head(K)
        docs = chosen["text"].fillna("").astype(str).tolist()

        prompt = build_prompt(q, docs)
        pred = llm_answer(prompt)
        prf = compute_f1(pred, gold)

        results.append({
            "query": q,
            "gold": gold,
            "pred": pred,
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
        })

    out_df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR, f"baseline_kshot_{K}_{order}.csv.gz")
    out_df.to_csv(out_csv, index=False, compression="gzip")
    print("Saved:", out_csv)

    print("Micro Precision:", out_df["precision"].mean())
    print("Micro Recall:   ", out_df["recall"].mean())
    P = out_df["precision"].mean()
    R = out_df["recall"].mean()
    micro_f1 = 0.0 if (P+R)==0 else 2*P*R/(P+R)
    print("Micro F1:       ", micro_f1)

if __name__ == "__main__":
    run_kshot("asc")
    run_kshot("desc")

