# Zero-Shot
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from torch.nn import functional as F
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

df = pd.read_csv("test_query_top20_candidates.csv")
print("Total rows:", len(df))

# Auth token and model setup
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

# Prompt setup
prompt_prefix = "Your task is to judge whether the sentiment of a movie review is positive or negative.\n"

def predict_label(query_text):
    prompt = f"{prompt_prefix}Review: {query_text}\nSentiment: "
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    last_token_logits = logits[:, -1, :]

    class_tokens = [
        tokenizer.encode("negative", add_special_tokens=False)[0],
        tokenizer.encode("positive", add_special_tokens=False)[0]
    ]

    probs = F.softmax(last_token_logits[:, class_tokens], dim=-1)
    return torch.argmax(probs).item()  # 0 = negative, 1 = positive

# Use only first candidate per query (0-shot)
grouped = df.groupby("query").first().reset_index()

results = []

for _, row in tqdm(grouped.iterrows(), total=len(grouped)):
    query = row["query"]
    gold = row["query_label"]
    pred = predict_label(query)

    results.append({
        "query": query,
        "query_label": gold,
        "predicted_label": pred,
        "correct": int(pred == gold)
    })

# Save CSV
result_df = pd.DataFrame(results)
result_df.to_csv("zeroshot_predictions.csv", index=False)

# Print evaluation
y_true = result_df["query_label"].tolist()
y_pred = result_df["predicted_label"].tolist()

report = classification_report(y_true, y_pred, target_names=["negative", "positive"], digits=4)
print("Zero-shot ICL Evaluation:\n")
print(report)
