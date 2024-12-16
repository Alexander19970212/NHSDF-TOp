import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np

def collate_fn_rs_loss(batch):
    """Custom collate function for the RSLossDataset"""
    # Extract points and targets from the batch
    points = torch.stack([item['grid'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])

    return points, targets

class RSLossConvDataset(Dataset):
    def __init__(self, root_dir, shape_type, grid_size=64, transform=None):
        """
        Args:
            root_dir (str): Directory containing the dataset
            shape_type (str): 'triangle' or 'quadrangle'
            grid_size (int): Size of the 2D grid (default: 64x64)
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = os.path.join(root_dir, shape_type)
        self.transform = transform
        self.grid_size = grid_size
        
        # Get list of all unique sample IDs
        self.sample_ids = []
        for file in os.listdir(self.root_dir):
            if file.endswith('_points.csv'):
                self.sample_ids.append(file.split('_points.csv')[0])
                
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_id = self.sample_ids[idx]
        
        # Load points and target
        points_path = os.path.join(self.root_dir, f'{sample_id}_points.csv')
        target_path = os.path.join(self.root_dir, f'{sample_id}_target.csv')
        
        # Read CSV files
        points_df = pd.read_csv(points_path)
        target_df = pd.read_csv(target_path)
        
        # Convert points to 2D grid
        grid = self._points_to_grid(points_df[['x', 'y', 'z']].values)

        # Apply random augmentations
        # 50% chance of transpose
        if torch.rand(1) > 0.5:
            grid = grid.transpose(1, 2)
            
        # 50% chance of horizontal flip
        if torch.rand(1) > 0.5:
            grid = torch.flip(grid, [2])
            
        # 50% chance of vertical flip  
        if torch.rand(1) > 0.5:
            grid = torch.flip(grid, [1])
        
        # Convert target to tensor
        target = torch.tensor(target_df['desired_angle_offset_sum'].values, dtype=torch.float32)
        
        sample = {
            'grid': grid,
            'target': target
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def _points_to_grid(self, points):
        grid = torch.zeros((1, self.grid_size, self.grid_size), dtype=torch.float32)
        
        # Normalize coordinates to [0, 1] range
        points_norm = points.copy()
        points_min = points.min(axis=0)
        points_max = points.max(axis=0)
        points_norm[:, :2] = (points[:, :2] - points_min[:2]) / (points_max[:2] - points_min[:2])

        # print(points_norm.min(), points_norm.max())
        
        # Map to grid indices
        indices = (points_norm[:, :2] * (self.grid_size - 1)).astype(int)
        
        # Create a sparse matrix to handle multiple points at same location
        sparse_grid = {}
        for idx, z in zip(indices, points[:, 2]):
            i, j = tuple(idx)
            if (i, j) in sparse_grid:
                sparse_grid[(i, j)].append(z)
            else:
                sparse_grid[(i, j)] = [z]
                
        # Fill grid with average z values
        for (i, j), z_values in sparse_grid.items():
            grid[0, i, j] = sum(z_values) / len(z_values)
            
        return grid