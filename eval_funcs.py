import numpy as np
import pandas as pd
from model import NCF
from typing import Tuple, Dict, Any
from preprocessing_utils import *
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import random
import wandb
import torch.nn as nn
import torch.nn.functional as F
from dataprep import *
from eval_funcs import *
import random
import gradio as gr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_actual_top_k_games(user_id, user_item_df, user_mapping, k=5):
    
    user_data = user_item_df[user_item_df['user_idx'] == user_id]
    top_k_actual = user_data.sort_values(by='playtime_forever', ascending=False).head(k)
    return top_k_actual['item_name'].tolist()

def get_predicted_top_k_games(model, user_id, user_mapping, item_mapping, user_item_df, k=5):
    model.eval()
    
    user_idx = torch.LongTensor([user_mapping[user_id]] * len(item_mapping)).to(device)    
    # Convert item_mapping values to a tensor and move to the same device as the model
    all_item_idxs = torch.LongTensor(list(item_mapping.values())).to(device)

    # Make predictions
    with torch.no_grad():
        predictions = model(user_idx, all_item_idxs).cpu().numpy().flatten()

    # Extract top-k items
    top_k_indices = predictions.argsort()[-k:][::-1]
    top_k_item_idxs = [list(item_mapping.values())[i] for i in top_k_indices]

    top_k_item_names = [user_item_df.loc[user_item_df['item_idx'] == idx, 'item_name'].iloc[0] for idx in top_k_item_idxs]

    return top_k_item_names

def precision_at_k(y_true, y_pred, k, threshold=0.5):
    # Sort by predicted score and take top k
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    top_k_true = y_true[top_k_indices]
    
    # Count number of true positives in top k (playtime above threshold)
    true_positives = np.sum(top_k_true > threshold)
    
    return true_positives / k

def predict_for_random_user(model, user_mapping, item_mapping, user_item_df, k=5):
    # Randomly select a user ID
    random_user_id = random.choice(list(user_mapping.keys()))
    print(random_user_id)
    print(f"Making predictions for random user {random_user_id}")
    # Check if the random_user_id exists in the DataFrame
    print(random_user_id in user_item_df['user_id'].values)

    
    # Call the existing function to make predictions for this user
    top_k_games = get_predicted_top_k_games(model, random_user_id, user_mapping, item_mapping, user_item_df, k)
    
    return random_user_id, top_k_games

def recall_at_k(y_true, y_pred, k, threshold=0.5):
    # Sort by predicted score and take top k
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    top_k_true = y_true[top_k_indices]
    
    # Count number of true positives in top k (playtime above threshold)
    true_positives = np.sum(top_k_true > threshold)
    
    # Count the total number of actual positives (relevant items)
    total_actual_positives = np.sum(y_true > threshold)
    
    if total_actual_positives == 0:
        return 0
    
    return true_positives / total_actual_positives