#!/usr/bin/env python3
"""
Preprocess downloaded ERA5 data to match the format expected by UpscaleDataset.
This script converts the CDS downloaded data into the samples_YYYY.nc format.
"""

import xarray as xr
import numpy as np
import os
import random
from pathlib import Path

def preprocess_era5_data():
    """Preprocess downloaded ERA5 data into the expected format."""
    
    print("ERA5 Data Preprocessing for ClimateDiffuse")
    print("=" * 40)
    
    data_dir = Path("data")
    
    # Years to process
    years = [1953, 1954, 1955, 1956, 1957]
    
    # Variable mapping from CDS to expected names - ONLY TEMPERATURE
    var_mapping = {
        'VAR_2T': '2m_temperature'
        # 'VAR_10U': '10m_u_component_of_wind',  # COMMENTED OUT
        # 'VAR_10V': '10m_v_component_of_wind'   # COMMENTED OUT
    }
    
    for year in years:
        print(f"📅 Processing year {year}...")
        
        year_dir = data_dir / str(year)
        
        if not year_dir.exists():
            print(f"  ❌ Year directory {year_dir} not found!")
            continue
        
        # Collect all monthly data
        monthly_data = []
        
        for month in range(1, 13):
            print(f"  📆 Processing month {month:02d}...")
            
            # Load all variables for this month
            month_data = {}
            
            for var_code, var_name in var_mapping.items():
                var_file = year_dir / f"{var_code}_{year}{month:02d}.nc"
                
                if not var_file.exists():
                    print(f"    ❌ File {var_file} not found!")
                    continue
                
                try:
                    # Load the data
                    ds = xr.open_dataset(var_file)
                    
                    # Get the variable (should be the only data variable)
                    data_vars = list(ds.data_vars.keys())
                    if len(data_vars) != 1:
                        print(f"    ⚠️  Unexpected number of variables in {var_file}")
                        continue
                    
                    var_data = ds[data_vars[0]]
                    
                    # Rename to expected variable name
                    var_data = var_data.rename(var_code)
                    
                    month_data[var_code] = var_data
                    
                    print(f"    ✅ Loaded {var_code}")
                    
                except Exception as e:
                    print(f"    ❌ Error loading {var_file}: {e}")
                    continue
            
            if len(month_data) == 1:  # Only temperature variable loaded
                # Merge all variables for this month
                month_ds = xr.Dataset(month_data)
                monthly_data.append(month_ds)
                print(f"    ✅ Month {month:02d} processed")
            else:
                print(f"    ❌ Month {month:02d} incomplete, skipping")
        
        if not monthly_data:
            print(f"  ❌ No valid monthly data found for {year}")
            continue
        
        # Concatenate all months
        print(f"  🔗 Concatenating {len(monthly_data)} months...")
        print(f"  📊 Total data points to process: {sum(ds.sizes.get('time', 0) for ds in monthly_data)} time steps")
        
        try:
            # Try concatenation with explicit join parameter to avoid warning
            print(f"  ⏳ Starting concatenation (this may take a while for large datasets)...")
            annual_ds = xr.concat(monthly_data, dim='time', join='outer')
            print(f"  ✅ Concatenation completed successfully!")
        except Exception as e:
            print(f"  ⚠️  Concatenation failed with join='outer', trying 'inner': {e}")
            try:
                annual_ds = xr.concat(monthly_data, dim='time', join='inner')
                print(f"  ✅ Concatenation completed with join='inner'!")
            except Exception as e2:
                print(f"  ❌ Concatenation failed completely: {e2}")
                print(f"  🔧 Trying to fix coordinate alignment...")
                
                # Fix coordinate alignment by dropping problematic coordinates
                fixed_data = []
                for i, ds in enumerate(monthly_data):
                    print(f"    🔧 Cleaning dataset {i+1}/{len(monthly_data)}...")
                    # Keep only essential coordinates
                    ds_clean = ds.drop_vars([var for var in ds.coords if var not in ['time', 'latitude', 'longitude']], errors='ignore')
                    fixed_data.append(ds_clean)
                
                print(f"  ⏳ Retrying concatenation with cleaned data...")
                annual_ds = xr.concat(fixed_data, dim='time')
                print(f"  ✅ Concatenation completed with cleaned data!")
        
        # Subsample to reduce correlation (like the original preprocessing)
        print(f"  🎲 Subsampling data...")
        
        # Set random seed for reproducibility
        random.seed(year)
        
        # Get all time indices
        time_inds = np.arange(len(annual_ds.time), dtype=int)
        random.shuffle(time_inds)
        
        # Select 30 random time steps (simulating the original subsampling)
        # For a full year, we'll take more samples to have enough data
        n_samples = min(360, len(time_inds))  # Up to 360 samples (30 per month)
        selected_inds = time_inds[:n_samples]
        
        # Subsample the data
        subsampled_ds = annual_ds.isel(time=selected_inds)
        
        # Save the processed data
        output_file = data_dir / f"samples_{year}.nc"
        subsampled_ds.to_netcdf(output_file)
        
        print(f"  ✅ Saved {output_file}")
        print(f"  📊 Data shape: {subsampled_ds.sizes}")
        print()
    
    print("🎉 Preprocessing completed!")
    print()
    print("Created files:")
    for year in years:
        output_file = data_dir / f"samples_{year}.nc"
        if output_file.exists():
            print(f"  ✅ samples_{year}.nc")
        else:
            print(f"  ❌ samples_{year}.nc (failed)")
    
    print()
    print("You can now run train_minimal.ipynb!")

if __name__ == "__main__":
    preprocess_era5_data()
