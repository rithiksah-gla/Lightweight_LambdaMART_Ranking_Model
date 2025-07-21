import os
import csv
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KDTree

# Load SST2 test data
def load_sst2_data(path):
    examples = []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            text, label = line.strip().split('\t')
            examples.append({"text": text, "label": int(label)})
    return examples

test_data = load_sst2_data('test.tsv')
print(test_data[:5])
print("Total test examples:", len(test_data))

# Create SBERT Embeddings and KDTree
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [ex['text'] for ex in test_data]
embeddings = sbert_model.encode(texts, show_progress_bar=True)
tree = KDTree(embeddings)

# Load LLaMA model and tokenizer
model_name = 'meta-llama/llama-2-7b-hf'

model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    use_auth_token=hf_token
)
tokenizer = LlamaTokenizer.from_pretrained(
    model_name,
    padding_side="left",
    use_auth_token=hf_token
)
tokenizer.add_special_tokens({'pad_token': '<PAD>'})
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# LLM scoring
def score_candidate_llm(query_text, query_label, candidate_text, candidate_label, tokenizer, model, device, prompt_prefix):
    demonstration = f"Review: {candidate_text}\nSentiment: {'positive' if candidate_label else 'negative'}\n"
    prompt = f"{prompt_prefix}{demonstration}Review: {query_text}\nSentiment: "
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    last_token_logits = logits[:, -1, :]
    class_tokens = [
        tokenizer.encode("negative", add_special_tokens=False)[0],
        tokenizer.encode("positive", add_special_tokens=False)[0]
    ]
    probs = F.softmax(last_token_logits[:, class_tokens], dim=-1)
    return probs[0, query_label].item()

# Build test triplets
N = 10
all_triplets = []

for idx, query_ex in tqdm(enumerate(test_data), total=len(test_data)):
    query_text = query_ex['text']
    query_label = query_ex['label']
    query_embedding = sbert_model.encode([query_text])
    dists, indices = tree.query(query_embedding, k=N+1)
    indices = indices[0]

    candidate_indices = [i for i in indices if i != idx][:N]

    for i, cand_idx in enumerate(candidate_indices):
        cand_ex = test_data[cand_idx]
        candidate_text = cand_ex['text']
        candidate_label = cand_ex['label']
        distance = float(dists[0][i + 1])

        prob = score_candidate_llm(
            query_text, query_label,
            candidate_text, candidate_label,
            tokenizer, model, device,
            prompt_prefix='Your task is to judge whether the sentiment of a movie review is positive or negative.\n'
        )

        all_triplets.append({
            'query': query_text,
            'query_label': query_label,
            'candidate': candidate_text,
            'candidate_label': candidate_label,
            'distance': distance,
            'score': prob
        })

# Save as CSV and TSV
fieldnames = ['query', 'query_label', 'candidate', 'candidate_label', 'distance', 'score']
csv_file = "test_dataset_lambdamart.csv"
tsv_file = "test_dataset_lambdamart.tsv"

with open(csv_file, "w", encoding="utf-8", newline="") as f_csv:
    writer_csv = csv.DictWriter(f_csv, fieldnames=fieldnames)
    writer_csv.writeheader()
    writer_csv.writerows(all_triplets)
print(f"Saved: {csv_file}")

with open(tsv_file, "w", encoding="utf-8", newline="") as f_tsv:
    writer_tsv = csv.DictWriter(f_tsv, fieldnames=fieldnames, delimiter='\t')
    writer_tsv.writeheader()
    writer_tsv.writerows(all_triplets)
print(f"Saved: {tsv_file}")
