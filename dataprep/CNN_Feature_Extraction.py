import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataprep.utils import get_img, all_img_paths
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Parameters
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "../models/Food/VGG16/Epoch=14- Loss=0.97 - val_acc = 0.59.h5")
DATASET_NAME = "Food"
ARCH_NAME = "VGG16"

# Load Image Paths
dataset_path = os.path.join(script_dir, "../Datasets", DATASET_NAME)
image_paths, labels = all_img_paths(dataset_path)

# Split Into Train and Test
train_paths, test_paths, train_labels, test_labels = train_test_split(
    np.array(image_paths), labels, test_size=0.1, random_state=42, shuffle=False)

# Load Classification Model
model = tf.keras.models.load_model(MODEL_PATH)
model.layers[-1].activation = tf.keras.activations.linear

# Select layer to extract features from (last Conv2D layer)
conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
feature_layer = conv_layers[-1].name
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer(feature_layer).output)

# Create directory to save features
feature_dir = os.path.join(script_dir, "../Features", DATASET_NAME, ARCH_NAME)
os.makedirs(feature_dir, exist_ok=True)

# Extract and save features for train images
for i, img_path in enumerate(train_paths):
    print(f'Processing image {i+1}/{len(train_paths)}: {img_path}')
    img = get_img(img_path)
    img_input = np.expand_dims(img, axis=0)
    features = feature_extractor.predict(img_input)
    # Normalize features to [0,1]
    features_min = features.min()
    features_max = features.max()
    features_norm = (features - features_min) / (features_max - features_min + 1e-8)
    # Save features as numpy array
    np.save(os.path.join(feature_dir, f'{i}.npy'), features_norm)
    print(f'Extracted features for image {i+1}/{len(train_paths)}', end='\r')

print("Feature extraction completed.")
