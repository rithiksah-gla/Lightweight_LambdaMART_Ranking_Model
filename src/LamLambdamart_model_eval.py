
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import ndcg_score
import numpy as np

# Load the test dataset
test_df = pd.read_csv("test_dataset_lambdamart.csv")

# Assign group_id for each query (assuming sorted queries)
def assign_group_ids(df):
    df = df.copy()
    df['group_id'] = None
    current_id = 0
    prev_query = None
    for idx, row in df.iterrows():
        if row['query'] != prev_query:
            current_id += 1
            prev_query = row['query']
        df.at[idx, 'group_id'] = current_id
    return df

test_df = assign_group_ids(test_df)

# Extract features and ground truth labels
X_test = test_df[['distance', 'score']]
test_df['relevance'] = (test_df['query_label'] == test_df['candidate_label']).astype(int)

# Load trained model
model = lgb.Booster(model_file="lambdamart_model.txt")

# Predict scores
test_df['predicted_score'] = model.predict(X_test)

# Evaluate NDCG@5 and NDCG@10 for each group
ndcg_at_5_list = []
ndcg_at_10_list = []

for _, group in test_df.groupby('group_id'):
    true_relevance = [group['relevance'].values]
    predicted_scores = [group['predicted_score'].values]

    if len(group) >= 5:
        ndcg_at_5 = ndcg_score(true_relevance, predicted_scores, k=5)
        ndcg_at_5_list.append(ndcg_at_5)

    if len(group) >= 10:
        ndcg_at_10 = ndcg_score(true_relevance, predicted_scores, k=10)
        ndcg_at_10_list.append(ndcg_at_10)

# Print average scores
print(f"Average NDCG@5: {np.mean(ndcg_at_5_list):.4f}")
print(f"Average NDCG@10: {np.mean(ndcg_at_10_list):.4f}")
