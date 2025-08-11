import torch
import pandas as pd
import numpy as np
import lightgbm as lgb
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report
from utils import extract_features, extract_lda_features, extract_drmm_features, get_idf_dict, build_lda_model

# Load retrieved top-50 SBERT candidates
df = pd.read_csv("/kaggle/input/top50-sbert-precompute/test_query_top50_48_features.csv")

# Build IDF dictionary and LDA model
texts = df["query"].tolist() + df["candidate"].tolist()
idf_dict = get_idf_dict(texts)
lda_model, lda_dict = build_lda_model(texts)

# Load LLaMA model and tokenizer
hf_token = "hf***"  # replace with your actual HF token
model_name = "meta-llama/llama-2-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_name, token=hf_token, padding_side="left")
tokenizer.add_special_tokens({'pad_token': '<PAD>'})

model_llm = LlamaForCausalLM.from_pretrained(model_name, token=hf_token, torch_dtype=torch.float16)
model_llm.resize_token_embeddings(len(tokenizer))
model_llm.config.pad_token_id = tokenizer.pad_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_llm.to(device)
model_llm.eval()

top_k = 10
prompt_prefix = "Your task is to judge whether the sentiment of a movie review is positive or negative.\n"

def predict_label_llm(prompt):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        logits = model_llm(**inputs).logits
    last_token_logits = logits[:, -1, :]
    class_tokens = [
        tokenizer.encode("negative", add_special_tokens=False)[0],
        tokenizer.encode("positive", add_special_tokens=False)[0]
    ]
    probs = F.softmax(last_token_logits[:, class_tokens], dim=-1)
    return int(torch.argmax(probs).item())

def run_kshot_letor_inference(order="desc"):
    model_path = "/kaggle/input/trained-models/lambdamart_with_48_features_asc.txt" if order == "asc" else "/kaggle/input/trained-models/lambdamart_with_48_features_desc.txt"
    model = lgb.Booster(model_file=model_path)

    y_true, y_pred, queries = [], [], []
    grouped = df.groupby("test_query_id")

    for test_query_id, group in tqdm(grouped, total=len(grouped), desc=f"K-shot LeToR ({order})"):
        query = group.iloc[0]["query"]
        query_label = int(group.iloc[0]["query_label"])
        queries.append(query)

        features = []
        for _, row in group.iterrows():
            candidate = row["candidate"]
            feats = (
                extract_features(query, candidate, idf_dict) +
                extract_lda_features(query, candidate, lda_model, lda_dict) +
                extract_drmm_features(query, candidate, lda_model, lda_dict)
            )
            features.append(feats)

        scores = model.predict(np.array(features))
        group = group.copy()
        group["lgbm_score"] = scores

        top_10 = group.sort_values("lgbm_score", ascending=False).head(top_k)
        top_examples = top_10.sort_values("lgbm_score", ascending=True) if order == "asc" else top_10

        prompt = prompt_prefix
        for _, row in top_examples.iterrows():
            sentiment = "positive" if row["candidate_label"] == 1 else "negative"
            prompt += f"Review: {row['candidate']}\nSentiment: {sentiment}\n"
        prompt += f"Review: {query}\nSentiment:"

        pred_label = predict_label_llm(prompt)
        y_true.append(query_label)
        y_pred.append(pred_label)

    pd.DataFrame({
        "query": queries,
        "query_label": y_true,
        "predicted_label": y_pred
    }).to_csv(f"kshot_letor_predictions_{order}.csv", index=False)

    print(f"\nK-shot LeToR ICL Evaluation ({order.upper()}):")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"], digits=4))

# Run both orders
run_kshot_letor_inference(order="asc")
run_kshot_letor_inference(order="desc")
