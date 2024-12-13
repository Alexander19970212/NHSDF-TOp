import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class SdfDataset(Dataset):
    """Custom Dataset for SDF data of multiple shapes"""
    def __init__(self, csv_files):
        # Load and combine data from multiple CSV files
        dfs = []
        feature_len = 0
        x_i = 0
        self.x_names = ['point_x', 'point_y', 'class']
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)
            feature_len += len(df.columns) - 3
            for col in df.columns:
                if col not in self.x_names and col != "sdf":
                    self.x_names.append(col)
                x_i += 1    

        self.n_classes = len(dfs)
        for class_i, df in enumerate(dfs):
            if self.n_classes == 1:
                df["class"] = 0
            else:
                df["class"] = class_i/(self.n_classes-1)

        self.data = pd.concat(dfs, ignore_index=True)
        # Replace NaN values with 0
        self.data = self.data.fillna(0)

        self.feature_dim = len(self.x_names)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Input features: point coordinates and shape parameters
        X_list = []
        for x_name in self.x_names:
            # if x_name != 'class':
            X_list.append(row[x_name])
            # else:
            #     if self.n_classes == 1:
            #         X_list.append(1)
            #     else:
            #         X_list.append(row[x_name]/(self.n_classes-1))

        X = torch.tensor(X_list, dtype=torch.float32)
        y = torch.tensor(row['sdf'], dtype=torch.float32)
        
        return X, y
    
#################################################################################################################

class SdfDatasetSurface(Dataset):
    """Custom Dataset for SDF data of multiple shapes"""
    def __init__(self, csv_files, cut_value=True, value_limit=0.001):
        # Load and combine data from multiple CSV files
        dfs = []
        points_dfs = []
        feature_len = 0
        x_i = 0
        self.x_names = ['class']
        for file in csv_files:
            df = pd.read_csv(f'{file}.csv')
            df_points = pd.read_csv(f'{file}_grid.csv')
            points_array = df_points.to_numpy()
            points_array = points_array.reshape(-1, 2)
            points_dfs.append(points_array)
            dfs.append(df)
            feature_len += len(df.columns) - 3
            for col in df.columns:
                if col not in self.x_names and col != "sdf":
                    self.x_names.append(col)
                x_i += 1    

        self.n_classes = len(dfs)
        for class_i, df in enumerate(dfs):
            if self.n_classes == 1:
                df["class"] = 0
            else:
                df["class"] = class_i/(self.n_classes-1)

        self.data = pd.concat(dfs, ignore_index=True)
        self.points_data = torch.tensor(np.array(points_dfs), dtype=torch.float32)
        # Replace NaN values with 0
        self.data = self.data.fillna(0)

        # Transform 'sdf' column from string to list of floats
        # print(self.data['sdf'].iloc[0].strip('[]').replace('\n', '').split(',')[0].split(' '))
        self.data['sdf'] = self.data['sdf'].apply(lambda x: list(map(float, x.strip('[]').replace('\n', '').split(','))))
        # print(self.data['sdf'])
        self.feature_dim = len(self.x_names) + 2

        self.value_limit = value_limit
        self.cut_value = cut_value

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Input features: shape parameters excluding point_x and point_y
        X_list = [row[x_name] for x_name in self.x_names if x_name not in ['point_x', 'point_y']]
        X = torch.tensor(X_list, dtype=torch.float32)
        # print(row['sdf'])
        y = torch.tensor(row['sdf'], dtype=torch.float32)
        class_idx = int(row['class'] * (self.n_classes - 1))
        points = self.points_data[class_idx]

        if self.cut_value:
            limit_mask = torch.logical_and(y > self.value_limit, y < (1-self.value_limit))
            y = y[limit_mask]
            points = points[limit_mask]

        sample = {
            'X': X,
            'y': y,
            'grid': points
        }

        return sample
    
def collate_fn_surface(batch):
    """Custom collate function for the SdfDatasetSurface"""
    # Extract points and targets from the batch
    points = [item['grid'] for item in batch]
    targets = [item['y'] for item in batch]
    X = [item['X'] for item in batch]
    return X, targets, points
    
