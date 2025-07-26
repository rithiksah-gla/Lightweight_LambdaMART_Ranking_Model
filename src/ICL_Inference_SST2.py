import torch
import pandas as pd
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.nn import functional as F
import re
from tqdm import tqdm
from sklearn.metrics import classification_report

# Load ranked candidates
df = pd.read_csv("test_query_top20_candidates.csv")
top_k = 10

# Load LLaMA
hf_token = "hf******"
model_name = 'meta-llama/llama-2-7b-hf'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
tokenizer.add_special_tokens({'pad_token': '<PAD>'})
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_auth_token=hf_token)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)
model.eval()

# Inference function
def predict_label_llm(prompt, tokenizer, model, device):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    last_token_logits = logits[:, -1, :]

    class_tokens = [
        tokenizer.encode("negative", add_special_tokens=False)[0],
        tokenizer.encode("positive", add_special_tokens=False)[0]
    ]
    probs = F.softmax(last_token_logits[:, class_tokens], dim=-1)
    return int(torch.argmax(probs).item())

# Perform ICL inference
y_true, y_pred = [], []

grouped = df.groupby("test_query_id")
for test_query_id, group in tqdm(grouped, total=len(grouped)):
    query = group.iloc[0]["query"]
    query_label = group.iloc[0]["query_label"]

    # Top-K candidates
    top_k_examples = group.sort_values("rank").head(top_k)

    # Build ICL prompt
    prompt = "Your task is to judge whether the sentiment of a movie review is positive or negative.\n"
    for _, row in top_k_examples.iterrows():
        demo = f"Review: {row['candidate']}\nSentiment: {'positive' if row['candidate_label'] else 'negative'}\n"
        prompt += demo
    prompt += f"Review: {query}\nSentiment: "

    pred_label = predict_label_llm(prompt, tokenizer, model, device)

    y_true.append(query_label)
    y_pred.append(pred_label)

# Save predictions
result_df = pd.DataFrame({
    "query": grouped.first()["query"],
    "query_label": y_true,
    "predicted_label": y_pred
})
result_df.to_csv("icl_predictions.csv", index=False)

# Print metrics
print("\nFinal ICL Evaluation:")
print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))
