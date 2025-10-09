import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataprep.utils import all_img_paths

script_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_NAME = "Food"
dataset_path = os.path.join(script_dir, "../Datasets", DATASET_NAME, "training")
print(f"Dataset path: {dataset_path}")
print(f"Path exists: {os.path.exists(dataset_path)}")
import glob
glob_pattern = os.path.join(dataset_path, "*", "*", "*")
print(f"Glob pattern: {glob_pattern}")
glob_result = glob.glob(glob_pattern)
print(f"Glob result length: {len(glob_result)}")
print(f"Sample glob: {glob_result[:5]}")

image_paths, labels = all_img_paths(dataset_path)
print(f"Number of images found: {len(image_paths)}")
print(f"Sample image paths: {image_paths[:5]}")
print(f"Sample labels: {labels[:5]}")
