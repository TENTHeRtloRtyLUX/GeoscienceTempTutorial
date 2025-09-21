#!/usr/bin/env python3
"""
Download ERA5 data using CDS API for ClimateDiffuse training.
This script downloads the exact data needed for train_minimal.ipynb.
"""

import cdsapi
import os
import time
from pathlib import Path

def download_era5_data():
    """Download ERA5 data for the years 1953-1957."""
    
    print("ERA5 Data Download for ClimateDiffuse")
    print("=" * 40)
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Initialize CDS client
    try:
        c = cdsapi.Client()
        print("✅ CDS API client initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing CDS API: {e}")
        print("Make sure you have set up your .cdsapirc file correctly!")
        return False
    
    # Years needed for train_minimal.ipynb
    years = [1953, 1954, 1955, 1956, 1957]
    
    # Variables to download - ONLY TEMPERATURE
    variables = {
        '2m_temperature': 'VAR_2T'
        # '10m_u_component_of_wind': 'VAR_10U',  # COMMENTED OUT
        # '10m_v_component_of_wind': 'VAR_10V'   # COMMENTED OUT
    }
    
    # Geographic area for US (same as in DatasetUS.py)
    area = [54.5, 233.6, 22.6, 297.5]  # North, West, South, East
    
    print(f"Downloading data for years: {years}")
    print(f"Geographic area: {area}")
    print()
    
    for year in years:
        print(f"📅 Processing year {year}...")
        
        # Create year directory
        year_dir = data_dir / str(year)
        year_dir.mkdir(exist_ok=True)
        
        for month in range(1, 13):
            print(f"  📆 Month {month:02d}...")
            
            # Determine days in month
            if month in [4, 6, 9, 11]:
                days = list(range(1, 31))  # 30 days
            elif month == 2:
                # Check for leap year
                if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                    days = list(range(1, 30))  # 29 days
                else:
                    days = list(range(1, 29))  # 28 days
            else:
                days = list(range(1, 32))  # 31 days
            
            # Download each variable
            for var_name, var_code in variables.items():
                output_file = year_dir / f"{var_code}_{year}{month:02d}.nc"
                
                if output_file.exists():
                    print(f"    ⏭️  {var_code} already exists, skipping...")
                    continue
                
                try:
                    print(f"    ⬇️  Downloading {var_code}...")
                    
                    c.retrieve(
                        'reanalysis-era5-single-levels',
                        {
                            'product_type': 'reanalysis',
                            'variable': var_name,
                            'year': str(year),
                            'month': f"{month:02d}",
                            'day': [f"{d:02d}" for d in days],
                            'time': [f"{h:02d}:00" for h in range(24)],
                            'area': area,
                            'format': 'netcdf',
                        },
                        str(output_file)
                    )
                    
                    print(f"    ✅ {var_code} downloaded successfully")
                    
                except Exception as e:
                    print(f"    ❌ Error downloading {var_code}: {e}")
                    continue
                
                # Small delay to be respectful to the server
                time.sleep(1)
        
        print(f"✅ Year {year} completed")
        print()
    
    print("🎉 Download process completed!")
    print(f"Data saved in: {data_dir.absolute()}")
    print()
    print("Next steps:")
    print("1. Run the preprocessing script to format the data")
    print("2. Test the data loading in train_minimal.ipynb")
    
    return True

if __name__ == "__main__":
    download_era5_data()
