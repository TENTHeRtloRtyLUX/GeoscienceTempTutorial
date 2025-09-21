#!/usr/bin/env python3
"""
Setup script for CDS API configuration.
This will help you create the .cdsapirc file needed for ERA5 data download.
"""

import os
from pathlib import Path

def setup_cds_api():
    """Setup CDS API configuration file."""
    
    print("CDS API Setup for ERA5 Data Download")
    print("=" * 40)
    print()
    print("To use the CDS API, you need to:")
    print("1. Register at https://cds.climate.copernicus.eu/user/register")
    print("2. Accept the terms of use for ERA5 data")
    print("3. Get your API key from your profile page")
    print()
    
    # Get user input
    api_key = input("Enter your CDS API key: ").strip()
    
    if not api_key:
        print("❌ API key is required!")
        return False
    
    # Create .cdsapirc content
    cdsapirc_content = f"""url: https://cds.climate.copernicus.eu/api
key: {api_key}
"""
    
    # Determine home directory
    home_dir = Path.home()
    cdsapirc_path = home_dir / ".cdsapirc"
    
    try:
        # Write the configuration file
        with open(cdsapirc_path, 'w') as f:
            f.write(cdsapirc_content)
        
        print(f"✅ Configuration saved to: {cdsapirc_path}")
        print("You can now use the CDS API to download ERA5 data!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating configuration file: {e}")
        return False

if __name__ == "__main__":
    setup_cds_api()
