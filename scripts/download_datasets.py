import os
import sys
from pathlib import Path
import requests
import zipfile
import logging
from tqdm import tqdm
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCISUMMNET_URL = "https://cs.stanford.edu/~myasu/projects/scisumm_net/scisummnet_release1.1__20190413.zip"
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCISUMMNET_DIR = DATA_DIR / "scisummnet"

def download_file(url: str, dest_path: Path) -> None:
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='B',
        unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)

def setup_datasets() -> None:
    """Download and setup required datasets"""
    try:
        # Create data directory
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download and extract ScisummNet if not present
        if not SCISUMMNET_DIR.exists():
            logger.info("Downloading ScisummNet dataset...")
            zip_path = DATA_DIR / "scisummnet.zip"
            
            # Download
            download_file(SCISUMMNET_URL, zip_path)
            
            # Extract
            logger.info("Extracting dataset...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(DATA_DIR)
            
            # Rename extracted directory
            extracted_dir = DATA_DIR / "scisummnet_release1.1__20190413"
            if extracted_dir.exists():
                shutil.move(str(extracted_dir), str(SCISUMMNET_DIR))
            
            # Cleanup
            zip_path.unlink()
            logger.info(f"ScisummNet dataset ready at {SCISUMMNET_DIR}")
        else:
            logger.info("ScisummNet dataset already exists")
            
    except Exception as e:
        logger.error(f"Error setting up datasets: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_datasets()