import os
import shutil
import random
import pandas as pd
from pathlib import Path

# --- Configuration Paths ---
# Use the paths you provided
XRAY_SOURCE_DIR = r"C:\Users\musty\Downloads\archive (3)\chest_xray"
SKIN_SOURCE_DIR = r"C:\Users\musty\Downloads\archive (4)"

# Target directory in your project
PROJECT_ROOT = Path(r"c:\Users\musty\OneDrive\Desktop\final  year project")
TARGET_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images"

# --- Mapping Configuration ---
# 1. Chest X-Rays
XRAY_MAPPING = {
    "Pneumonia": "train/PNEUMONIA",
    "Common Cold": "train/NORMAL",  # Using Normal X-rays as a proxy for healthy lungs/common cold
}
MAX_XRAYS_PER_CLASS = 3000

# 2. Skin Lesions (HAM10000)
# We map these to diseases that actually exist in your 41-class tabular dataset
# for academic demonstration of the multi-modal pipeline.
SKIN_MAPPING = {
    "nv": "Acne",               # Nevi mapped to Acne
    "mel": "Fungal infection",  # Melanoma mapped to Fungal infection
    "bcc": "Psoriasis",         # Basal cell carcinoma mapped to Psoriasis
    "bkl": "Impetigo"           # Benign keratosis mapped to Impetigo
}
MAX_SKIN_PER_CLASS = 2000

def ingest_xrays():
    print("=== Ingesting Chest X-Rays ===")
    for target_class, rel_source in XRAY_MAPPING.items():
        source_dir = Path(XRAY_SOURCE_DIR) / rel_source
        target_dir = TARGET_IMAGE_DIR / "xray" / target_class
        
        if not source_dir.exists():
            print(f"⚠️ Source directory not found: {source_dir}")
            continue
            
        target_dir.mkdir(parents=True, exist_ok=True)
        
        all_images = list(source_dir.glob("*.jpeg")) + list(source_dir.glob("*.jpg"))
        random.shuffle(all_images)
        selected_images = all_images[:MAX_XRAYS_PER_CLASS]
        
        print(f"Copying {len(selected_images)} images for {target_class}...")
        for img_path in selected_images:
            shutil.copy2(img_path, target_dir / img_path.name)

def ingest_skin():
    print("\n=== Ingesting Skin Lesions ===")
    metadata_path = Path(SKIN_SOURCE_DIR) / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        print(f"⚠️ Metadata not found: {metadata_path}")
        return

    df = pd.read_csv(metadata_path)
    
    # Locate image folders
    img_dir_1 = Path(SKIN_SOURCE_DIR) / "HAM10000_images_part_1"
    img_dir_2 = Path(SKIN_SOURCE_DIR) / "HAM10000_images_part_2"
    
    # Track counts to enforce MAX_SKIN_PER_CLASS
    counts = {k: 0 for k in SKIN_MAPPING.values()}
    
    # Shuffle dataframe to get random subset
    df = df.sample(frac=1).reset_index(drop=True)
    
    for _, row in df.iterrows():
        dx = row['dx']
        if dx not in SKIN_MAPPING:
            continue
            
        target_class = SKIN_MAPPING[dx]
        if counts[target_class] >= MAX_SKIN_PER_CLASS:
            continue
            
        img_id = row['image_id'] + ".jpg"
        
        # Check which folder it's in
        src_img = img_dir_1 / img_id
        if not src_img.exists():
            src_img = img_dir_2 / img_id
            if not src_img.exists():
                continue
                
        target_dir = TARGET_IMAGE_DIR / "skin" / target_class
        target_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(src_img, target_dir / img_id)
        counts[target_class] += 1
        
        if sum(counts.values()) >= len(SKIN_MAPPING) * MAX_SKIN_PER_CLASS:
            break
            
    for cls, count in counts.items():
        print(f"Copied {count} images for {cls}")

if __name__ == "__main__":
    print("Starting Multi-Modal Image Ingestion...\n")
    ingest_xrays()
    ingest_skin()
    print("\n[SUCCESS] Image ingestion complete! Your dataset is ready for True Multi-Modal Training.")
