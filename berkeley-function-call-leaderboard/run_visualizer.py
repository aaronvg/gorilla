#!/usr/bin/env python3
"""
Quick launcher for the BFCL Results Visualization Tool

This script uses uv to run the Streamlit app with all required dependencies.
"""

import subprocess
import sys
import os

def main():
    # Check if uv is available
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: uv is not installed. Please install uv first:")
        print("curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
    
    # Dependencies required for the visualization
    deps = [
        "streamlit>=1.28.0",
        "plotly>=5.15.0", 
        "pandas>=2.0.0"
    ]
    
    # Build the uv run command
    cmd = ["uv", "run"]
    for dep in deps:
        cmd.extend(["--with", dep])
    
    cmd.extend(["streamlit", "run", "visualize_bfcl_results.py"])
    
    print("Starting BFCL Results Visualization Tool...")
    print(f"Running: {' '.join(cmd)}")
    print()
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run the command
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error running visualization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()