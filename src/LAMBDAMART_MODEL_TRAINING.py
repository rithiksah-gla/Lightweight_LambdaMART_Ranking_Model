import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from utils import tokenize, extract_features, get_idf_dict

# Load data
train_df = pd.read_csv("train_dataset_lambdamart_N50.csv")
val_df = pd.read_csv("validation_dataset_lambdamart_N50.csv")

# Create IDF dictionary from training queries and docs
train_texts = train_df["query"].tolist() + train_df["candidate"].tolist()
idf_dict = get_idf_dict(train_texts)

# Feature names
feature_names = [
    'num_q_terms', 'num_q_unique', 'num_d_terms', 'num_d_unique',
    'min_q_idf', 'max_q_idf', 'sum_q_idf',
    'min_d_idf', 'max_d_idf', 'sum_d_idf',
    'overlap', 'bm25_score'
]

# Extract features
train_feats = train_df.apply(lambda row: extract_features(row['query'], row['candidate'], idf_dict), axis=1, result_type='expand')
train_feats.columns = feature_names
train_df = pd.concat([train_df, train_feats], axis=1)

val_feats = val_df.apply(lambda row: extract_features(row['query'], row['candidate'], idf_dict), axis=1, result_type='expand')
val_feats.columns = feature_names
val_df = pd.concat([val_df, val_feats], axis=1)

# Assign relevance labels properly
def assign_relevance(df):
    df = df.sort_values(by=['query', 'score'], ascending=[True, False])
    df['relevance'] = df.groupby('query').cumcount(ascending=False)
    return df

train_df = assign_relevance(train_df)
val_df = assign_relevance(val_df)

# Prepare training data
X_train = train_df[feature_names]
y_train = train_df['relevance']
group_train = train_df.groupby('query').size().to_list()

X_val = val_df[feature_names]
y_val = val_df['relevance']
group_val = val_df.groupby('query').size().to_list()

train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)

# LightGBM LambdaMART parameters
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5],
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbosity': -1,
    'label_gain': list(range(max(y_train.max(), y_val.max()) + 1))
}

# Train with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    valid_names=["train", "val"],
    callbacks=[
        early_stopping(stopping_rounds=20),
        log_evaluation(period=10)
    ]
)

# Save the model
model.save_model("lambdamart_model.txt")
