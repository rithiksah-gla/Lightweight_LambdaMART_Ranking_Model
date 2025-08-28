# baseline_zero_shot_from_csv.py
import os
import pandas as pd
import torch
from tqdm import tqdm
from nq_utils_1 import tokenizer, model, device, compute_f1, sanitize_query

TEST_PATH = "/scratch/2980356s/data/SST2/output/nq_open_top15_bm25_llm_pairs/nq_open_top15_bm25_llm_pairs_test.csv.gz"
OUT_CSV   = "./baselines/baseline_zero_shot_results.csv.gz"

PROMPT_INSTRUCTION = (
    "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) "
    "from one of the provided documents. If none of the documents contain the answer, respond with NO-$"
)
MAX_NEW_TOKENS = 20

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

def zero_shot_answer(question: str) -> str:
    # Same prompt; “Documents:” section intentionally empty for zero-shot
    prompt = f"{PROMPT_INSTRUCTION}\nDocuments:\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, num_beams=1)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # robust tail extraction
    tail = decoded.split("Answer:")[-1] if "Answer:" in decoded else decoded
    return tail.strip()

def main():
    df = pd.read_csv(TEST_PATH, compression="gzip")

    # One row per query group
    results = []
    for query, group in tqdm(df.groupby("query"), desc="Zero-shot"):
        q = sanitize_query(str(query))
        # ground truth is a list stored as string; eval-safe via literal_eval if needed
        gt = group.iloc[0]["ground_truth"]
        try:
            gold = eval(gt) if isinstance(gt, str) else gt
        except Exception:
            gold = [str(gt)]

        pred = zero_shot_answer(q)
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
    out_df.to_csv(OUT_CSV, index=False, compression="gzip")
    print("Saved:", OUT_CSV)

    # micro-averaged P/R/F1 over all queries (sum TP/FP/FN through compute_f1 definition is token-based;
    # here we report mean of per-query metrics for simplicity & comparability)
    print("Micro Precision:", out_df["precision"].mean())
    print("Micro Recall:   ", out_df["recall"].mean())
    P = out_df["precision"].mean()
    R = out_df["recall"].mean()
    micro_f1 = 0.0 if (P+R)==0 else 2*P*R/(P+R)
    print("Micro F1:       ", micro_f1)

if __name__ == "__main__":
    main()
