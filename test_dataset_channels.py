#!/usr/bin/env python3

import sys
sys.path.append('src/')
import torch
from DatasetUS import UpscaleDataset

# Test the dataset to see what channels it provides
print("Testing dataset channel configuration...")

try:
    # Create dataset
    dataset = UpscaleDataset('data/', year_start=1953, year_end=1953, 
                           constant_variables=['lsm', 'z'])
    
    # Get a batch
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)))
    
    print(f"Input shape: {batch['inputs'].shape}")
    print(f"Target shape: {batch['targets'].shape}")
    print(f"Number of input channels: {batch['inputs'].shape[1]}")
    print(f"Dataset varnames: {dataset.varnames}")
    print(f"Dataset n_var: {dataset.n_var}")
    print(f"Constant variables: ['lsm', 'z']")
    print(f"Expected total channels: {dataset.n_var + 2}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
