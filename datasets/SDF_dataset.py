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
    def __init__(self, csv_files, exclude_ellipse=False):
        # Load and combine data from multiple CSV files
        self.max_radius_sum = 6
        dfs = []
        feature_len = 0
        x_i = 0
        self.x_names = ['point_x', 'point_y', 'class']
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)
            feature_len += len(df.columns) - 3
            for col in df.columns:
                if col not in self.x_names and col not in ["sdf", "arc_ratio"]:
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

        if exclude_ellipse:
            self.data = self.data[self.data["class"] != 0]

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

        X = torch.tensor(X_list, dtype=torch.float32)
        arc_ratio = torch.tensor(row['arc_ratio'], dtype=torch.float32)
        y = torch.tensor(row['sdf'], dtype=torch.float32)
        
        return X, y, arc_ratio
    
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
                if col not in self.x_names and col not in ["sdf", "arc_ratio"]:
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
    
class RadiusDataset(Dataset):
    """Custom Dataset for SDF data of multiple shapes"""
    def __init__(self, csv_files, exclude_ellipse=False):
        # Load and combine data from multiple CSV files
        feature_names = ['feature_id',
            'point_x', 'point_y', 'class', 'semi_axes_ratio', 'v1_x', 'v1_y', 
            'r_t1', 'r_t2', 'r_t3', 'v3_x', 'v3_y', 'v4_x', 'v4_y', 
            'r_q1', 'r_q2', 'r_q3', 'r_q4', 'arc_ratio'
        ]
        self.max_radius_sum = 6
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)    

        self.n_classes = 3
        for class_i, df in enumerate(dfs):
            if self.n_classes == 1:
                df["class"] = 0
            else:
                df["class"] = (class_i+1)/(self.n_classes-1)

        self.data = pd.concat(dfs, ignore_index=True)
        self.data = self.data.reindex(columns=feature_names, fill_value=0)
        # Replace NaN values with 0
        self.data = self.data.fillna(0)
        geom_feature_ids = self.data['feature_id'].unique()
        self.geom_feature_ids = np.unique(geom_feature_ids)

        self.feature_dim = len(feature_names) - 2

    def __len__(self):
        return len(self.geom_feature_ids)
    
    def __getitem__(self, idx):
        geom_feature_id = self.geom_feature_ids[idx]
        row = self.data[self.data['feature_id'] == geom_feature_id]
        X_row = row.drop(columns=['feature_id', 'arc_ratio'])

        X = torch.tensor(X_row.values, dtype=torch.float32)
        arc_ratio = torch.tensor(row['arc_ratio'].values, dtype=torch.float32)
        
        return X, 0, arc_ratio
    
class ReconstructionDataset(Dataset):
    """Custom Dataset for SDF data of multiple shapes"""
    def __init__(self, csv_files):
        # Load and combine data from multiple CSV files
        feature_names = [
            'point_x', 'point_y', 'class', 'semi_axes_ratio', 'v1_x', 'v1_y', 
            'r_t1', 'r_t2', 'r_t3', 'v3_x', 'v3_y', 'v4_x', 'v4_y', 
            'r_q1', 'r_q2', 'r_q3', 'r_q4', 'arc_ratio'
        ]
        self.max_radius_sum = 6
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)    

        self.n_classes = 3
        for class_i, df in enumerate(dfs):
            if self.n_classes == 1:
                df["class"] = 0
            else:
                df["class"] = class_i/(self.n_classes-1)

        self.data = pd.concat(dfs, ignore_index=True)
        self.data = self.data.reindex(columns=feature_names, fill_value=0)
        # Replace NaN values with 0
        self.data = self.data.fillna(0)

        self.feature_dim = len(feature_names) - 1

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        X_row = row.drop(labels=['arc_ratio'])

        X = torch.tensor(X_row.values, dtype=torch.float32)
        arc_ratio = torch.tensor(row['arc_ratio'].item(), dtype=torch.float32)
        
        return X, torch.tensor(0, dtype=torch.float32), arc_ratio
    
class Dataset3DHeavisideSDF(Dataset):
    """Custom Dataset for SDF data of multiple shapes"""
    def __init__(self, csv_dir, index_list_csv):
        import os
        from collections import OrderedDict

        self.csv_dir = csv_dir
        # List all CSV files in the directory, sorted in natural order.
        self.csv_files = sorted([os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')])
        self.rows_per_file = 2000  # Each CSV file contains 2000 rows

        # Load the index list containing global row indices (ranging from 0 to 10000*2000 - 1)
        dataset_index_list = pd.read_csv(index_list_csv)
        self.indices = dataset_index_list['index'].values.astype(np.int64)

        # print(self.indices)

        # Precompute mapping from each global index to (file index, row index within file)
        self.file_indices = self.indices // self.rows_per_file
        self.row_indices = self.indices % self.rows_per_file

        self.feature_names = [
            'point_x', 'point_y', 'point_z',
            'v3_x', 'v3_y',
            'v4_x', 'v4_y',
            'r_q1', 'r_q2', 'r_q3', 'r_q4',  # q means quadrangle
            'z_level',
            'heaviside_sdf',
            'arc_ratio'
        ]

        self.feature_dim = len(self.feature_names) - 2
        # print(self.feature_dim)
        
        # Initialize an LRU cache to hold a limited number of loaded CSV files.
        self._cache = OrderedDict()
        self._cache_max_size = 100

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Retrieve precomputed file and row indices for the given dataset index.
        file_idx = int(self.file_indices[idx])
        row_idx = int(self.row_indices[idx])
        if file_idx >= len(self.csv_files):
            raise IndexError("Global index out of range of available CSV files.")
        file_path = self.csv_files[file_idx]

        # Use cached CSV if available; otherwise, load from disk.
        if file_path not in self._cache:
            df = pd.read_csv(file_path)
            # Ensure the dataframe has the expected columns; fill missing ones with 0.
            df = df.reindex(columns=self.feature_names, fill_value=0)
            if len(self._cache) >= self._cache_max_size:
                self._cache.popitem(last=False)  # Remove the least recently used file.
            self._cache[file_path] = df
        else:
            # Move this file to the end to mark it as recently used.
            df = self._cache.pop(file_path)
            self._cache[file_path] = df

        try:
            row = df.iloc[row_idx]
        except:
            print(row_idx)
            print(df.shape)
            print(df.columns)
            print(df.index)
            # print(df.head())  
            raise IndexError("Global index out of range of available CSV files.")
        # Separate the 'arc_ratio' target from the remaining features.
        arc_ratio = torch.tensor(row['arc_ratio'], dtype=torch.float32)
        X = torch.tensor(row.drop(labels=['arc_ratio', 'heaviside_sdf']).values, dtype=torch.float32)

        # X = torch.tensor(, dtype=torch.float32)
        # arc_ratio = torch.tensor(row['arc_ratio'], dtype=torch.float32)
        y = torch.tensor(row['heaviside_sdf'], dtype=torch.float32)

        return X, y, arc_ratio


def collate_fn_surface(batch):
    """Custom collate function for the SdfDatasetSurface"""
    # Extract points and targets from the batch
    points = [item['grid'] for item in batch]
    targets = [item['y'] for item in batch]
    X = [item['X'] for item in batch]
    return X, targets, points


    
