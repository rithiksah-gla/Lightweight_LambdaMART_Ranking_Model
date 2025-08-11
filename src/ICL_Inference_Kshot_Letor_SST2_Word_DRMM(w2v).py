# icl_letor_32_w2v.py
import torch
import pandas as pd
import numpy as np
import lightgbm as lgb
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report
#from utils import extract_features, extract_drmm_w2v_features, get_idf_dict, build_w2v_model

# Precompute file from step 2
df = pd.read_csv("/kaggle/working/test_query_top50_word_DRMM_w2v_features.csv")

# Build IDF and W2V from train.tsv (same corpus choice as training)
texts = df["query"].tolist() + df["candidate"].tolist()
idf_dict  = get_idf_dict(texts)
w2v_model = build_w2v_model(texts)

# LLaMA
hf_token = "hf***"
model_name = "meta-llama/llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name, token=hf_token, padding_side="left")
tokenizer.add_special_tokens({'pad_token':'<PAD>'})
model_llm = LlamaForCausalLM.from_pretrained(model_name, token=hf_token, torch_dtype=torch.float16)
model_llm.resize_token_embeddings(len(tokenizer))
model_llm.config.pad_token_id = tokenizer.pad_token_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_llm.to(device).eval()

top_k = 10
prompt_prefix = "Your task is to judge whether the sentiment of a movie review is positive or negative.\n"

def predict_label_llm(prompt):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        logits = model_llm(**inputs).logits
    last = logits[:, -1, :]
    class_tokens = [
        tokenizer.encode("negative", add_special_tokens=False)[0],
        tokenizer.encode("positive", add_special_tokens=False)[0],
    ]
    probs = F.softmax(last[:, class_tokens], dim=-1)
    return int(torch.argmax(probs).item())

def run_kshot_letor_inference(order="desc"):
    model_path = "/kaggle/working/lambdamart_with_32_w2v_features_desc.txt"
    model = lgb.Booster(model_file=model_path)

    y_true, y_pred, queries = [], [], []
    grouped = df.groupby("test_query_id")

    for qid, group in tqdm(grouped, total=len(grouped), desc=f"K-shot LeToR ({order})"):
        query = group.iloc[0]["query"]; q_label = int(group.iloc[0]["query_label"])
        queries.append(query)

        feats = []
        for _, row in group.iterrows():
            cand = row["candidate"]
            feats.append(extract_features(query, cand, idf_dict) + extract_drmm_w2v_features(query, cand, w2v_model))

        scores = model.predict(np.array(feats))
        g = group.copy(); g["lgbm_score"] = scores
        top_10 = g.sort_values("lgbm_score", ascending=False).head(top_k)
        top_examples = top_10.sort_values("lgbm_score", ascending=True) if order=="asc" else top_10

        prompt = prompt_prefix
        for _, r in top_examples.iterrows():
            sentiment = "positive" if r["candidate_label"] == 1 else "negative"
            prompt += f"Review: {r['candidate']}\nSentiment: {sentiment}\n"
        prompt += f"Review: {query}\nSentiment:"

        pred = predict_label_llm(prompt)
        y_true.append(q_label); y_pred.append(pred)

    pd.DataFrame({"query":queries, "query_label":y_true, "predicted_label":y_pred}).to_csv(
        f"kshot_letor_predictions_{order}_32_w2v.csv", index=False
    )
    print(f"\nK-shot LeToR ICL Evaluation ({order.upper()}):")
    print(classification_report(y_true, y_pred, target_names=["negative","positive"], digits=4))

run_kshot_letor_inference("asc")
run_kshot_letor_inference("desc")
