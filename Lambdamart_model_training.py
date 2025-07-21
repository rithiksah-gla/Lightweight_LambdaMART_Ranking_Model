import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# Load training and validation datasets
train_df = pd.read_csv("train_dataset_lambdamart.csv")
val_df = pd.read_csv("validation_dataset_lambdamart.csv")

# Encode labels (positive -> 1, negative -> 0) if not already binary
label_encoder = LabelEncoder()
train_df['query_label'] = label_encoder.fit_transform(train_df['query_label'])
train_df['candidate_label'] = label_encoder.transform(train_df['candidate_label'])
val_df['query_label'] = label_encoder.transform(val_df['query_label'])
val_df['candidate_label'] = label_encoder.transform(val_df['candidate_label'])

# Assign group ID (qid) to each query based on 'query' field
def assign_group_ids(df):
    df = df.copy()
    df['qid'] = None
    current_qid = 0
    prev_query = None
    for idx, row in df.iterrows():
        if row['query'] != prev_query:
            current_qid += 1
            prev_query = row['query']
        df.at[idx, 'qid'] = current_qid
    return df

train_df = assign_group_ids(train_df)
val_df = assign_group_ids(val_df)

# Extract features — you can add more later
def extract_features(df):
    return df[['distance', 'score']]

X_train = extract_features(train_df)
y_train = (train_df['query_label'] == train_df['candidate_label']).astype(int)
group_train = train_df.groupby('qid').size().to_numpy()

X_val = extract_features(val_df)
y_val = (val_df['query_label'] == val_df['candidate_label']).astype(int)
group_val = val_df.groupby('qid').size().to_numpy()

# LightGBM Datasets
train_set = lgb.Dataset(X_train, label=y_train, group=group_train)
val_set = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_set)

# LambdaMART parameters
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'boosting': 'gbdt',
    'ndcg_eval_at': [5, 10],
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'verbose': -1
}

# Train model
print("Training LambdaMART model...")
model = lgb.train(
    params,
    train_set,
    valid_sets=[train_set, val_set],
    valid_names=['train', 'valid'],
    num_boost_round=1000
)

# Save model
model.save_model("lambdamart_model.txt")
print("Model saved to lambdamart_model.txt")