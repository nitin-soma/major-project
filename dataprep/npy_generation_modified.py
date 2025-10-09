import sys
sys.path.append('dataprep')
from utils import all_img_paths
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np

DATASET_NAME = "Food"
ARCH_NAME = "VGG16"

# Load image paths and labels
import os
dataset_path = os.path.join("Datasets", DATASET_NAME, "training")
abs_dataset_path = os.path.abspath(dataset_path)
print(f"Dataset path: {dataset_path}")
print(f"Absolute dataset path: {abs_dataset_path}")
print(f"Path exists: {os.path.exists(abs_dataset_path)}")

image_paths, labels = all_img_paths(abs_dataset_path)
print(f"Number of image paths found: {len(image_paths)}")
print(f"Number of labels found: {len(labels)}")

# Convert labels to int
unique_labels = sorted(list(set(labels)))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
int_labels = [label_to_int[label] for label in labels]

# Split into train and test
train_paths, test_paths, train_labels, test_labels = train_test_split(
    np.array(image_paths), int_labels, test_size=0.1, random_state=42, shuffle=False)

# Create data list
data = []
for i, img_path in enumerate(train_paths):
    # Load input image
    inp = cv2.imread(img_path)
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(inp, (64, 64))

    # Load CNN features
    feature_path = os.path.join("Features", DATASET_NAME, ARCH_NAME, f"{i}.npy")
    features = np.load(feature_path)

    # For modified version, out is the input image (self-supervised)
    out = inp.copy()

    # Append to data
    tup = [inp, features, out, train_labels[i]]
    data.append(tup)
    print(f"Processed {i+1}/{len(train_paths)}", end='\r')

data = np.array(data, dtype=object)
np.random.seed(7)
np.random.shuffle(data)

# Save to npys/
os.makedirs("../npys", exist_ok=True)
try:
    np.save(f'../npys/{DATASET_NAME}_{ARCH_NAME}.npy', data)
    print("NPY file saved successfully.")
except Exception as e:
    print(f"Error saving NPY file: {e}")
