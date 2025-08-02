import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import matplotlib.pyplot as plt
#from utils import extract_features, extract_lda_features, get_idf_dict, build_lda_model

# Load data
train_df = pd.read_csv("/kaggle/input/llm-utility-n50/train_dataset_lambdamart_N50.csv")
val_df = pd.read_csv("/kaggle/input/llm-utility-n50/validation_dataset_lambdamart_N50.csv")

# Create IDF dictionary from training queries and docs
train_texts = train_df["query"].tolist() + train_df["candidate"].tolist()
idf_dict = get_idf_dict(train_texts)

# Train LDA model
print("Training LDA model...")
lda_model, lda_dict = build_lda_model(train_texts)

# Define 28 feature names (12 handcrafted + 16 LDA)
handcrafted_features = [
    'num_q_terms', 'num_q_unique', 'num_d_terms', 'num_d_unique',
    'min_q_idf', 'max_q_idf', 'sum_q_idf',
    'min_d_idf', 'max_d_idf', 'sum_d_idf',
    'overlap', 'bm25_score'
]

lda_features = [
    'q_avg_phi_norm', 'q_phi_norm_min', 'q_phi_norm_max',
    'q_phi_pair_min_sim', 'q_phi_pair_max_sim', 'q_phi_pair_avg_sim',
    'd_avg_phi_norm', 'd_phi_norm_min', 'd_phi_norm_max',
    'd_phi_pair_min_sim', 'd_phi_pair_max_sim', 'd_phi_pair_avg_sim',
    'qd_phi_single_link_sim', 'qd_phi_complete_link_sim', 'qd_phi_avg_link_sim',
    'qd_theta_cos_sim'
]

feature_names = handcrafted_features + lda_features

# Feature extraction function
def extract_all_features(row):
    handcrafted = extract_features(row['query'], row['candidate'], idf_dict)
    lda_feats = extract_lda_features(row['query'], row['candidate'], lda_model, lda_dict)
    return handcrafted + lda_feats

# Extract features for training set
print("Extracting features from training set...")
train_feature_matrix = train_df.apply(extract_all_features, axis=1, result_type='expand')
train_feature_matrix.columns = feature_names
train_df = pd.concat([train_df, train_feature_matrix], axis=1)

# Extract features for validation set
print("Extracting features from validation set...")
val_feature_matrix = val_df.apply(extract_all_features, axis=1, result_type='expand')
val_feature_matrix.columns = feature_names
val_df = pd.concat([val_df, val_feature_matrix], axis=1)

# Assign relevance labels by rank
def assign_relevance(df):
    df = df.sort_values(by=['query', 'score'], ascending=[True, False])
    df['relevance'] = df.groupby('query').cumcount(ascending=False)
    return df

train_df = assign_relevance(train_df)
val_df = assign_relevance(val_df)

# Prepare LightGBM input
X_train = train_df[feature_names]
y_train = train_df['relevance']
group_train = train_df.groupby('query').size().to_list()

X_val = val_df[feature_names]
y_val = val_df['relevance']
group_val = val_df.groupby('query').size().to_list()

train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)

# Training params
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5],
    'learning_rate': 0.01,
    'num_leaves': 31,
    'verbosity': -1,
    'label_gain': list(range(max(y_train.max(), y_val.max()) + 1))
}

# Track metrics
evals_result = {}

# Train model
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    valid_names=["train", "val"],
    callbacks=[
        early_stopping(stopping_rounds=20),
        log_evaluation(period=10),
        lgb.record_evaluation(evals_result)
    ],
)

# Save model
model.save_model("lambdamart_with_28_features.txt")

# Plot NDCG@5
plt.figure(figsize=(10, 6))
plt.plot(evals_result["train"]["ndcg@5"], label="Train NDCG@5")
plt.plot(evals_result["val"]["ndcg@5"], label="Val NDCG@5")
plt.xlabel("Iteration")
plt.ylabel("NDCG@5")
plt.title("NDCG@5 over Iterations (LambdaMART with 28 features)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
