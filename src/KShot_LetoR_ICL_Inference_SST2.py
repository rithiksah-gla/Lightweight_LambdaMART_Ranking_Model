import torch
import pandas as pd
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report

# Load top-20 candidates ranked by LambdaMART
df = pd.read_csv("/kaggle/input/20-top-test-query/test_query_top20_candidates.csv")
top_k = 10

# Load LLaMA model and tokenizer
hf_token = "hf****"
model_name = 'meta-llama/llama-2-7b-hf'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = LlamaTokenizer.from_pretrained(model_name, token=hf_token, padding_side="left")
tokenizer.add_special_tokens({'pad_token': '<PAD>'})

model = LlamaForCausalLM.from_pretrained(model_name, token=hf_token, torch_dtype=torch.float16)
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

# Run K-shot inference
y_true, y_pred, queries = [], [], []

grouped = df.groupby("test_query_id")
for test_query_id, group in tqdm(grouped, total=len(grouped)):
    query = group.iloc[0]["query"]
    query_label = group.iloc[0]["query_label"]
    queries.append(query)

    # Top-K from LambdaMART
    top_k_examples = group.sort_values("rank").head(top_k)

    # Prompt construction
    prompt = "Your task is to judge whether the sentiment of a movie review is positive or negative.\n"
    for _, row in top_k_examples.iterrows():
        demo = f"Review: {row['candidate']}\nSentiment: {'positive' if row['candidate_label'] else 'negative'}\n"
        prompt += demo
    prompt += f"Review: {query}\nSentiment: "

    pred_label = predict_label_llm(prompt, tokenizer, model, device)

    y_true.append(query_label)
    y_pred.append(pred_label)

# Save results
result_df = pd.DataFrame({
    "query": queries,
    "query_label": y_true,
    "predicted_label": y_pred
})
result_df.to_csv("kshot_letor_predictions.csv", index=False)

# Print 4-decimal classification report
print("\nK-shot LeToR ICL Evaluation (k=10):")
print(classification_report(y_true, y_pred, target_names=["negative", "positive"], digits=4))
