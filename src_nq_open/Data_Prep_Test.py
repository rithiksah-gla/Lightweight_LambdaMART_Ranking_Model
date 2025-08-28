import re
import os
import pandas as pd
import pyterrier as pt
from datasets import load_dataset
from typing import List, Dict
from nq_utils import predict_answer, compute_f1, sanitize_query, tokenizer, device, model
from tqdm import tqdm
import torch
import os
import string

# Set PyTorch CUDA memory allocation configuration to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# --------------------------
# Config
# --------------------------
index_dir = "/scratch/2980356s/data/SST2/wiki2018_nqopen"
top_k = 15  # Updated to N=15
out_prefix = f"nq_open_top{top_k}_bm25_llm_pairs"
output_dir = "./output"  # Default output directory
hf_nq_ds = "florin-hf/nq_open_gold"
splits_to_try = ["test"]

# --------------------------
# Init PyTerrier + open index
# --------------------------
if not pt.java.started():
    pt.java.init()

index = pt.IndexFactory.of(index_dir)
br = pt.terrier.Retriever(index, wmodel="BM25", metadata=["docno", "text"])

for split in splits_to_try:
    try:
        print(f"\n=== Loading NQ-Open split: {split} ===")
        ds = load_dataset(hf_nq_ds, split=split)
    except Exception as e:
        print(f"Skipping split '{split}' (not available): {e}")
        continue

    questions = [r["question"] for r in ds]
    answers = []  # Use answer as ground truth
    for r in ds:
        if "answers" in r:
            if isinstance(r["answers"], (list, tuple)):
                answers.append(r["answers"])
            elif isinstance(r["answers"], dict) and "text" in r["answers"]:
                answers.append(r["answers"]["text"])
            else:
                answers.append([str(r["answers"])])
        else:
            answers.append([str(r.get("answer", ""))])

    qry_df = pd.DataFrame({"qid": range(len(questions)), "query": questions})
    q = qry_df["query"].fillna("").astype(str).apply(sanitize_query)
    qry_df["query"] = q
    print(f"Retrieving top-{top_k} for {len(qry_df)} queries...")
    res = br.transform(qry_df).groupby("qid").head(top_k).reset_index(drop=True)

    if res.empty:
        print("Warning: No results retrieved, skipping prediction.")
        continue

    qid2q = dict(zip(qry_df["qid"], qry_df["query"]))
    res["query"] = res["qid"].map(qid2q)

    print("Predicting answers and computing F1 scores...")
    pr_list, rc_list, f1_list = [], [], []
    llm_answers = []  # Store LLM predictions

    task_instruction = "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-$"

    # Clear GPU memory
    torch.cuda.empty_cache()

    for idx, row in tqdm(res.iterrows(), total=len(res)):
        query = row["query"]
        retrieved_text = row["text"]
        ground_truth = answers[int(row["qid"])]  # Ground truth from answer field
        prompt = f"{task_instruction}\nDocuments:\nDocument: {retrieved_text.lower()}\nQuestion: {query.lower()}\nAnswer:"
        
        # Debug: Print inputs
        print(f"Query: {query}")
        print(f"Retrieved Text: {retrieved_text[:50]}...")
        print(f"Ground Truth: {ground_truth}")

        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            try:
                outputs = model.generate(**inputs, max_new_tokens=20, num_beams=1, early_stopping=False)
                # Check for NaNs or infs in outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print("Warning: NaNs or infs detected in outputs, skipping")
                    pred_answer = "NO-RES"
                else:
                    pred_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
            except RuntimeError as e:
                print(f"Generation error: {e}, skipping")
                pred_answer = "NO-RES"
        
        llm_answers.append(pred_answer)
        prf = compute_f1(pred_answer, ground_truth)  # F1 between LLM answer and ground truth answer
        pr_list.append(prf["precision"])
        rc_list.append(prf["recall"])
        f1_list.append(prf["f1"])

    res["llm_precision"] = pr_list
    res["llm_recall"] = rc_list
    res["llm_f1"] = f1_list
    res["llm_answer"] = llm_answers  # Add LLM answer column
    res["ground_truth"] = [answers[int(qid)] for qid in res["qid"]]  # Add ground truth column

    cols = ["qid", "query", "rank", "score", "text", "llm_precision", "llm_recall", "llm_f1", "llm_answer", "ground_truth"]  # Removed docno
    out = res[cols].sort_values(["qid", "rank"]).reset_index(drop=True)

    # Ensure output directory exists
    output_path = os.path.join(output_dir, out_prefix)
    try:
        os.makedirs(output_path, exist_ok=True)
        out_csv = os.path.join(output_path, f"{out_prefix}_{split}.csv.gz")
        out_tsv = os.path.join(output_path, f"{out_prefix}_{split}.tsv.gz")
        out.to_csv(out_csv, index=False, compression="gzip")
        out.to_csv(out_tsv, index=False, sep="\t", compression="gzip")
        print(f"Saved: {out_csv} ({len(out)} rows)")
        print(f"Saved: {out_tsv} ({len(out)} rows)")
    except (PermissionError, IOError) as e:
        print(f"Error saving files: {e}, check directory permissions or available space.")

    os.system("rm -rf ~/.cache/huggingface/*")

print("\nAll done.")
