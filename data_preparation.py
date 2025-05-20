# data_processing.py
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

def prepare_data(X, y, n_splits=10, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return skf.split(X, y)

def prepare_splits(data, y, n_splits=10, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    masks = []

    for train_idx, test_idx in skf.split(data.x.numpy(), y):
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        masks.append((train_mask, test_mask))

    return masks