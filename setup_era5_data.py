#!/usr/bin/env python3
"""
Complete setup script for ERA5 data download and preprocessing.
This script handles the entire process from CDS API setup to data preprocessing.
"""

import os
import sys
from pathlib import Path

def check_cds_api():
    """Check if CDS API is properly configured."""
    try:
        import cdsapi
        c = cdsapi.Client()
        print("✅ CDS API is properly configured")
        return True
    except Exception as e:
        print(f"❌ CDS API configuration issue: {e}")
        print()
        print("To fix this:")
        print("1. Register at https://cds.climate.copernicus.eu/user/register")
        print("2. Accept the terms of use for ERA5 data")
        print("3. Get your API key from your profile page")
        print("4. Run: python setup_cds_api.py")
        return False

def main():
    """Main setup function."""
    
    print("ClimateDiffuse ERA5 Data Setup")
    print("=" * 40)
    print()
    
    # Check if CDS API is configured
    if not check_cds_api():
        print("Please configure CDS API first, then run this script again.")
        return False
    
    print()
    print("This will download ERA5 data for years 1953-1957")
    print("This may take a while and will use significant bandwidth.")
    print()
    
    response = input("Do you want to continue? (y/N): ").strip().lower()
    if response != 'y':
        print("Setup cancelled.")
        return False
    
    print()
    print("Starting ERA5 data download...")
    print("=" * 40)
    
    # Run the download script
    try:
        import download_era5_cds
        success = download_era5_cds.download_era5_data()
        
        if not success:
            print("❌ Download failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during download: {e}")
        return False
    
    print()
    print("Starting data preprocessing...")
    print("=" * 40)
    
    # Run the preprocessing script
    try:
        import preprocess_era5_data
        preprocess_era5_data.preprocess_era5_data()
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return False
    
    print()
    print("🎉 Setup completed successfully!")
    print()
    print("Next steps:")
    print("1. Open examples/train_minimal.ipynb")
    print("2. Run all cells to test the training")
    print("3. The data should now load correctly!")
    
    return True

if __name__ == "__main__":
    main()

