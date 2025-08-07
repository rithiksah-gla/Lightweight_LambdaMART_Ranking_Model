import torch
import pandas as pd
import numpy as np
import lightgbm as lgb
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report
#from utils import extract_features, extract_lda_features, get_idf_dict, build_lda_model

# Load top-50 retrieved candidates from SBERT
df = pd.read_csv("/kaggle/working/test_query_top50_candidates_with_16_LDA_features.csv")

# Load trained LambdaMART model
model = lgb.Booster(model_file="/kaggle/working/lambdamart_with_28_features.txt")

# Extract IDF from all candidate/query texts for feature computation
idf_dict = get_idf_dict(df["query"].tolist() + df["candidate"].tolist())

# Build LDA model
print("Building LDA model...")
lda_model, lda_dict = build_lda_model(df["query"].tolist() + df["candidate"].tolist())

# Load LLaMA model + tokenizer
hf_token = "hf***"
model_name = "meta-llama/llama-2-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_name, token=hf_token, padding_side="left")
tokenizer.add_special_tokens({'pad_token': '<PAD>'})

model_llm = LlamaForCausalLM.from_pretrained(model_name, token=hf_token, torch_dtype=torch.float16)
model_llm.resize_token_embeddings(len(tokenizer))
model_llm.config.pad_token_id = tokenizer.pad_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_llm.to(device)
model_llm.eval()

# Parameters
top_k = 10
prompt_prefix = "Your task is to judge whether the sentiment of a movie review is positive or negative.\n"
feature_names = [
    'num_q_terms', 'num_q_unique', 'num_d_terms', 'num_d_unique',
    'min_q_idf', 'max_q_idf', 'sum_q_idf',
    'min_d_idf', 'max_d_idf', 'sum_d_idf',
    'overlap', 'bm25_score',
    'q_avg_phi', 'q_argmin_norm', 'q_argmax_norm',
    'q_phi_min_sim', 'q_phi_max_sim', 'q_phi_avg_sim',
    'd_avg_phi', 'd_argmin_norm', 'd_argmax_norm',
    'd_phi_min_sim', 'd_phi_max_sim', 'd_phi_avg_sim',
    'qd_single_link', 'qd_complete_link', 'qd_avg_link',
    'theta_cos_sim'
]

# Predict label using LLaMA
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

# Inference runner
def run_kshot_letor_inference(order="desc"):
    y_true, y_pred, queries = [], [], []

    grouped = df.groupby("test_query_id")
    for test_query_id, group in tqdm(grouped, total=len(grouped), desc=f"K-shot LeToR (LDA + handcrafted) [{order}]"):
        query = group.iloc[0]["query"]
        query_label = int(group.iloc[0]["query_label"])
        queries.append(query)

        # Feature extraction (handcrafted + LDA)
        features = []
        for _, row in group.iterrows():
            f_hand = extract_features(query, row["candidate"], idf_dict)
            f_lda = extract_lda_features(query, row["candidate"], lda_model, lda_dict)
            features.append(f_hand + f_lda)

        group = group.copy()
        group["lgbm_score"] = model.predict(np.array(features))

        # Sort by LambdaMART score
        top_examples = group.sort_values("lgbm_score", ascending=(order == "asc")).head(top_k)

        # Prompt
        prompt = prompt_prefix
        for _, row in top_examples.iterrows():
            sentiment = "positive" if row["candidate_label"] == 1 else "negative"
            prompt += f"Review: {row['candidate']}\nSentiment: {sentiment}\n"
        prompt += f"Review: {query}\nSentiment: "

        pred_label = predict_label_llm(prompt)

        y_true.append(query_label)
        y_pred.append(pred_label)

    # Save results
    result_df = pd.DataFrame({
        "query": queries,
        "query_label": y_true,
        "predicted_label": y_pred
    })
    result_df.to_csv(f"kshot_letor_lda_predictions_{order}.csv", index=False)

    print(f"\nK-shot LeToR ICL Evaluation (LDA+handcrafted, {order.upper()}):")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"], digits=4))

# Run for both orders
run_kshot_letor_inference(order="asc")
run_kshot_letor_inference(order="desc")
