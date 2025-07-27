import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from torch.nn import functional as F
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

# Load top-20 candidates per test query (from LambdaMART output)
df = pd.read_csv("test_query_top20_candidates.csv")
print("Total rows:", len(df))

# Group and sort by score descending
df = df.sort_values(by=["query", "score"], ascending=[True, False])
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
K = 10  # Number of demonstrations
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
    return torch.argmax(probs).item()  # 0 = negative, 1 = positive

results = []

for query, group in tqdm(groups, total=len(groups)):
    query_label = group.iloc[0]['query_label']
    
    # Top K demonstrations
    demonstrations = ""
    for i in range(min(K, len(group))):
        cand = group.iloc[i]
        sentiment = "positive" if cand['candidate_label'] == 1 else "negative"
        demonstrations += f"Review: {cand['candidate']}\nSentiment: {sentiment}\n"

    # Final prompt
    test_prompt = f"{prompt_prefix}{demonstrations}Review: {query}\nSentiment: "
    
    pred = predict_label(test_prompt)
    results.append({
        "query": query,
        "query_label": int(query_label),
        "predicted_label": pred,
        "correct": int(pred == int(query_label))
    })

# Save predictions
result_df = pd.DataFrame(results)
result_df.to_csv("kshot_predictions.csv", index=False)

# Print evaluation
y_true = result_df["query_label"].tolist()
y_pred = result_df["predicted_label"].tolist()

report = classification_report(
    y_true,
    y_pred,
    target_names=["negative", "positive"],
    digits=4
)

print("K-shot ICL Evaluation:\n")
print(report)
