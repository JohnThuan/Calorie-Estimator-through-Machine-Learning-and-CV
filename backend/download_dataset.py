"""
Download and extract Food-101 dataset
"""

import os
import requests
import tarfile
from tqdm import tqdm

DATASET_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
DATA_DIR = "data"
DATASET_FILE = "food-101.tar.gz"

def download_file(url, filename):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    print(f"Downloading {filename}...")
    print(f"Size: {total_size / (1024*1024):.2f} MB")
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            bar.update(len(chunk))
    
    print(f"[OK] Downloaded {filename}")

def extract_tar(filename, extract_path):
    """Extract tar.gz file"""
    print(f"Extracting {filename}...")
    
    with tarfile.open(filename, 'r:gz') as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc="Extracting") as pbar:
            for member in members:
                tar.extract(member, path=extract_path)
                pbar.update(1)
    
    print(f"[OK] Extracted to {extract_path}")

def main():
    """Download and extract Food-101 dataset"""
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    dataset_path = os.path.join(DATA_DIR, DATASET_FILE)
    
    # Download dataset if not exists
    if not os.path.exists(dataset_path):
        print("Downloading Food-101 dataset...")
        print("This is a large file (~5 GB), please be patient...")
        download_file(DATASET_URL, dataset_path)
    else:
        print(f"[OK] Dataset already downloaded: {dataset_path}")
    
    # Extract dataset
    food101_dir = os.path.join(DATA_DIR, "food-101")
    if not os.path.exists(food101_dir):
        print("\nExtracting dataset...")
        extract_tar(dataset_path, DATA_DIR)
    else:
        print(f"[OK] Dataset already extracted: {food101_dir}")
    
    print("\n[SUCCESS] Dataset ready!")
    print(f"Location: {os.path.abspath(food101_dir)}")
    print(f"Classes: 101 food categories")
    print(f"Images: 101,000 total (750 training + 250 test per class)")
    print("\nNext step: Run 'python train_model.py' to start training!")

if __name__ == "__main__":
    main()






