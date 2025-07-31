import csv
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KDTree
from utils import load_sst2_data, score_candidate_llm

# Load validation (dev) dataset
val_data = load_sst2_data('dev.tsv')
print(val_data[:5])
print("Total validation examples:", len(val_data))

# Create SBERT Embeddings and KDTree
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [ex['text'] for ex in val_data]
embeddings = sbert_model.encode(texts, show_progress_bar=True)
tree = KDTree(embeddings)

# Load LLaMA model and tokenizer
hf_token = "hf***"
model_name = 'meta-llama/llama-2-7b-hf'

model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    use_auth_token=hf_token,
    cache_dir="/scratch/2980356s/hf_cache/"
)
tokenizer = LlamaTokenizer.from_pretrained(
    model_name,
    padding_side="left",
    use_auth_token=hf_token,
    cache_dir="/scratch/2980356s/hf_cache/"
)
tokenizer.add_special_tokens({'pad_token': '<PAD>'})
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Build validation triplets
N = 50
all_triplets = []

for idx, query_ex in tqdm(enumerate(val_data), total=len(val_data)):
    query_text = query_ex['text']
    query_label = query_ex['label']
    query_embedding = sbert_model.encode([query_text])
    dists, indices = tree.query(query_embedding, k=N+1)
    indices = indices[0]

    candidate_indices = [i for i in indices if i != idx][:N]

    for i, cand_idx in enumerate(candidate_indices):
        cand_ex = val_data[cand_idx]
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
csv_file = "validation_dataset_lambdamart_N50.csv"
tsv_file = "validation_dataset_lambdamart_N50.tsv"

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
