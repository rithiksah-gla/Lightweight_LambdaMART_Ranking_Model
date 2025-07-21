import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Load training data
df = pd.read_csv("train_dataset_lambdamart.csv")

# Binary relevance: 1 if query_label == candidate_label, else 0
df['relevance'] = (df['query_label'] == df['candidate_label']).astype(int)

# Define Pairwise Dataset
class PairwiseDataset(Dataset):
    def __init__(self, df):
        self.pairs = []
        grouped = df.groupby("query")
        for query, group in grouped:
            positives = group[group['relevance'] == 1]
            negatives = group[group['relevance'] == 0]
            for _, pos in positives.iterrows():
                for _, neg in negatives.iterrows():
                    self.pairs.append((query, pos['candidate'], neg['candidate']))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

# Model definition
class BertRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, query, doc):
        # Encode [CLS] query [SEP] doc [SEP]
        encoded = tokenizer(query, doc, return_tensors='pt', padding=True, truncation=True, max_length=512)
        encoded = {k: v.to(self.bert.device) for k, v in encoded.items()}
        outputs = self.bert(**encoded)
        cls_output = outputs.last_hidden_state[:, 0, :]
        score = self.linear(cls_output)
        return score

# Initialize tokenizer, model, optimizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertRanker().cuda()
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.MarginRankingLoss(margin=1.0)

# Prepare dataloader
dataset = PairwiseDataset(df)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training loop
epochs = 3
model.train()
for epoch in range(epochs):
    total_loss = 0
    for query, pos_doc, neg_doc in tqdm(dataloader):
        optimizer.zero_grad()
        pos_scores = model(query, pos_doc).squeeze()
        neg_scores = model(query, neg_doc).squeeze()
        target = torch.ones(pos_scores.size()).to(model.bert.device)
        loss = loss_fn(pos_scores, neg_scores, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "bert_ranker.pt")
