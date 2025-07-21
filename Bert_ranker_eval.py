import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import pandas as pd
from transformers import BertModel
import torch.nn as nn

# Load the test dataset
df = pd.read_csv("test_dataset_lambdamart.csv")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define the same model class
class BertRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, query, doc):
        encoded = tokenizer(query, doc, return_tensors='pt', padding=True, truncation=True, max_length=512)
        encoded = {k: v.to(self.bert.device) for k, v in encoded.items()}
        outputs = self.bert(**encoded)
        cls_output = outputs.last_hidden_state[:, 0, :]
        score = self.linear(cls_output)
        return score

# Initialize and load trained weights
model = BertRanker()
model.load_state_dict(torch.load("bert_ranker.pt", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


# Group queries
grouped = df.groupby('query')

ndcg_5_scores = []
ndcg_10_scores = []

print("Evaluating...")

for query, group in tqdm(grouped, total=len(grouped)):
    candidate_texts = group['candidate'].tolist()
    true_relevance = (group['query_label'] == group['candidate_label']).astype(int).tolist()

    # Tokenize and move to device
    inputs = tokenizer(candidate_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

    # Get model scores
    with torch.no_grad():
        preds = model([query] * len(candidate_texts), candidate_texts).squeeze(-1)
        preds = preds.cpu().numpy()

    # Compute NDCG
    ndcg_5 = ndcg_score([true_relevance], [preds], k=5)
    ndcg_10 = ndcg_score([true_relevance], [preds], k=10)

    ndcg_5_scores.append(ndcg_5)
    ndcg_10_scores.append(ndcg_10)

# Report averages
print(f"Average NDCG@5: {sum(ndcg_5_scores) / len(ndcg_5_scores):.4f}")
print(f"Average NDCG@10: {sum(ndcg_10_scores) / len(ndcg_10_scores):.4f}")
