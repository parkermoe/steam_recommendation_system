from typing import Tuple, Dict, Any
from preprocessing_utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
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


def create_mappings(df, column):
    unique_ids = sorted(df[column].unique())  # Sorting added here
    print(f"Number of unique {column}: {len(unique_ids)}")
    id_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
    return id_to_idx

class SimplifiedSteamDataset(Dataset):
    def __init__(self, user_item_df, user_mapping, item_mapping):
        self.df = user_item_df.copy()
        
        self.df['user_idx'] = self.df['user_id'].map(user_mapping)
        self.df['item_idx'] = self.df['item_id'].map(item_mapping)
        
        print(f"Missing user_idx: {self.df['user_idx'].isna().sum()}")
        print(f"Missing item_idx: {self.df['item_idx'].isna().sum()}")
        
        # Scaling playtime
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.y = self.scaler.fit_transform(self.df['playtime_forever'].values.reshape(-1, 1)).flatten()
        
        self.user_idxs = self.df['user_idx'].values
        self.item_idxs = self.df['item_idx'].values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.y[idx], self.user_idxs[idx], self.item_idxs[idx]

def prepare_simple_data_loaders(user_item_path, batch_size=128):
    # Load data
    user_item_df = load_json_to_df(user_item_path)
    user_item_df.sort_values(by=['user_id', 'playtime_forever'], ascending=[True, False], inplace=True)
    
    # Drop duplicates based on 'user_id' and 'item_id'
    #user_item_df.drop_duplicates(subset=['user_id', 'item_id'], inplace=True)
    def test_mapping(mapping, unique_ids):
        # Test 1: Ensure every unique ID has a mapping
        for unique_id in unique_ids:
            assert unique_id in mapping, f"{unique_id} not found in mapping"

        # Test 2: Ensure reverse mapping is consistent
        reverse_mapping = {v: k for k, v in mapping.items()}
        for unique_id in unique_ids:
            idx = mapping[unique_id]
            assert reverse_mapping[idx] == unique_id, f"Inconsistent mapping for {unique_id}"
    
    # Create mappings
    user_mapping = create_mappings(user_item_df, 'user_id')
    item_mapping = create_mappings(user_item_df, 'item_id')

    test_mapping(user_mapping, user_item_df['user_id'].unique())
    test_mapping(item_mapping, user_item_df['item_id'].unique())

    # Create the final dataset using the new mappings
    full_dataset = SimplifiedSteamDataset(user_item_df, user_mapping, item_mapping)

    # Split into train, validation, and test sets
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size], 
    generator=torch.Generator().manual_seed(42)  # set the seed for reproducibility
    )

# Create a DataFrame from val_dataset
    val_data = [val_dataset[i] for i in range(len(val_dataset))]
    val_df = pd.DataFrame(val_data, columns=['playtime_scaled', 'user_idx', 'item_idx'])
    # Create DataLoader for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Extract the indices from val_dataset
    val_indices = val_dataset.indices

    # Create a validation DataFrame using these indices
    val_df = full_dataset.df.iloc[val_indices].reset_index(drop=True)

    return train_loader, val_loader, test_loader, user_mapping, item_mapping, full_dataset.df, val_df