import subprocess
import os

COLMAP_PATH = r'D:\colmap-main\bin\colmap.exe'

def run_colmap_sparse(image_dir, output_dir, database_path):
    os.makedirs(output_dir, exist_ok=True)
    
    # Feature extraction
    print("Running COLMAP feature extraction...")
    subprocess.run([
        COLMAP_PATH, 'feature_extractor',
        '--database_path', database_path,
        '--image_path', image_dir
    ], check=True)
    print("✓ Feature extraction completed")
    
    # Exhaustive matcher
    print("Running COLMAP exhaustive matching...")
    subprocess.run([
        COLMAP_PATH, 'exhaustive_matcher',
        '--database_path', database_path
    ], check=True)
    print("✓ Exhaustive matching completed")
    
    # Mapper
    print("Running COLMAP mapping...")
    subprocess.run([
        COLMAP_PATH, 'mapper',
        '--database_path', database_path,
        '--image_path', image_dir,
        '--output_path', output_dir
    ], check=True)
    print("✓ Mapping completed")

