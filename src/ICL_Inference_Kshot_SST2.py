import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from torch.nn import functional as F
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

# Load SBERT Top-50 candidate retrieval output
df = pd.read_csv("test_query_top50_candidates.csv")
print("Total rows:", len(df))

# Compute reciprocal score from distance
df["score"] = 1 / df["distance"]

# Group by test query
groups = df.groupby("query")

# HuggingFace auth token and model setup
hf_token = "hf***"
model_name = "meta-llama/llama-2-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_name, token=hf_token, padding_side="left")
tokenizer.add_special_tokens({'pad_token': '<PAD>'})

model = LlamaForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

device = model.device
K = 10
prompt_prefix = "Your task is to judge whether the sentiment of a movie review is positive or negative.\n"

def predict_label(prompt):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    last_token_logits = logits[:, -1, :]
    class_tokens = [
        tokenizer.encode("negative", add_special_tokens=False)[0],
        tokenizer.encode("positive", add_special_tokens=False)[0]
    ]
    probs = F.softmax(last_token_logits[:, class_tokens], dim=-1)
    return torch.argmax(probs).item()

# Run for both ASC and DESC on same top-10
for sort_order in ["asc", "desc"]:
    results = []

    print(f"\nRunning k-shot inference using top-{K} candidates sorted {sort_order.upper()}")

    for query, group in tqdm(groups, total=len(groups)):
        query_label = group.iloc[0]['query_label']

        # Step 1: pick top K candidates based on highest score (i.e., lowest distance)
        top_k_group = group.sort_values(by="score", ascending=False).head(K)

        # Step 2: reorder those top-K based on ASC or DESC score order
        if sort_order == "asc":
            top_k_group = top_k_group.sort_values(by="score", ascending=True)
        else:
            top_k_group = top_k_group.sort_values(by="score", ascending=False)

        # Step 3: create prompt
        demonstrations = ""
        for _, cand in top_k_group.iterrows():
            sentiment = "positive" if cand['candidate_label'] == 1 else "negative"
            demonstrations += f"Review: {cand['candidate']}\nSentiment: {sentiment}\n"

        test_prompt = f"{prompt_prefix}{demonstrations}Review: {query}\nSentiment:"
        pred = predict_label(test_prompt)

        results.append({
            "query": query,
            "query_label": int(query_label),
            "predicted_label": pred,
            "correct": int(pred == int(query_label))
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(f"kshot_predictions_1overdist_sorted_{sort_order}.csv", index=False)

    # Evaluation
    y_true = result_df["query_label"].tolist()
    y_pred = result_df["predicted_label"].tolist()

    report = classification_report(
        y_true,
        y_pred,
        target_names=["negative", "positive"],
        digits=4
    )

    print(f"\nEvaluation Report - K-Shot (1/distance) Sorted {sort_order.upper()}:\n")
    print(report)
