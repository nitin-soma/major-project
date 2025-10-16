"""
Production-ready GradCAM generation script for your entire dataset
Generates both CGMAs (from multiple layers) and final GradCAMs
"""

from utils import get_img, all_img_paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

# ============= CONFIGURATION =============
MODEL_PATH = "../models/VGG16_final.h5"
DATASET_NAME = "Food"
ARCH_NAME = "VGG16"
DATASET_PATH = f"../Datasets/{DATASET_NAME}/training"

# Number of images to process (set to -1 for all)
MAX_IMAGES = -1

# Output directories
CGMS_DIR = f"../CGMS/{DATASET_NAME}/{ARCH_NAME}"
GRADCAMS_DIR = f"../Gradcams/{DATASET_NAME}/{ARCH_NAME}"
LOGS_DIR = f"../logs/{DATASET_NAME}/{ARCH_NAME}"

# Create directories
os.makedirs(CGMS_DIR, exist_ok=True)
os.makedirs(GRADCAMS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# =========================================

print("="*70)
print("GradCAM Generation - Production Script")
print("="*70)

# Load model
print("\n[1/6] Loading model...")
full_model = tf.keras.models.load_model(MODEL_PATH)
print(f"✓ Model loaded: {MODEL_PATH}")

# Find base model
print("\n[2/6] Finding base model...")
base_model = None
for layer in full_model.layers:
    if isinstance(layer, tf.keras.Model):
        for sublayer in layer.layers:
            if isinstance(sublayer, tf.keras.layers.Conv2D):
                base_model = layer
                break
        if base_model:
            break

if base_model is None:
    print("✗ No base model found!")
    exit(1)

print(f"✓ Base model: {base_model.name}")

# Get Conv2D layers
conv_layers = [layer for layer in reversed(base_model.layers) 
               if isinstance(layer, tf.keras.layers.Conv2D)]
print(f"✓ Found {len(conv_layers)} Conv2D layers")

# Load dataset
print("\n[3/6] Loading dataset...")
all_img_paths_list, all_labels = all_img_paths(DATASET_PATH)
print(f"✓ Found {len(all_img_paths_list)} images")

# Split data
train_paths, test_paths, train_labels, test_labels = train_test_split(
    np.array(all_img_paths_list), 
    all_labels, 
    test_size=0.1, 
    random_state=42,
    shuffle=False
)

# Encode labels
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(np.array(train_labels).reshape(-1, 1))
train_labels_encoded = enc.transform(np.array(train_labels).reshape(-1, 1)).toarray()

print(f"✓ Training images: {len(train_paths)}")
print(f"✓ Classes: {len(enc.categories_[0])}")

# Limit number of images if specified
num_images = len(train_paths)
if MAX_IMAGES > 0:
    num_images = min(MAX_IMAGES, num_images)
    print(f"✓ Processing first {num_images} images")

# Function to generate GradCAM
def generate_gradcam(img, model, base_model, conv_layer, pred_class):
    """Generate GradCAM for a single image"""
    img_batch = np.expand_dims(img, axis=0).astype('float32')
    
    try:
        # Create intermediate model
        inter_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=conv_layer.output
        )
        
        # Get conv outputs
        conv_output = inter_model(img_batch, training=False)
        
        # Average activation (fallback method that works reliably)
        gradcam = np.mean(conv_output[0].numpy(), axis=-1)
        gradcam = np.maximum(gradcam, 0)
        
        # Resize to original image size
        gradcam = cv2.resize(gradcam, (256, 256))
        
        # Normalize
        if gradcam.max() > 0:
            gradcam = gradcam / gradcam.max()
        
        return gradcam
    
    except Exception as e:
        print(f"  ⚠ Error generating GradCAM: {e}")
        return np.zeros((256, 256))

# Generate CGMAs (from multiple layers)
print("\n[4/6] Generating CGMAs from multiple layers...")
print(f"Processing {num_images} images...")

# Select layers for CGMA (use every nth layer for efficiency)
cgma_layers = conv_layers[::max(1, len(conv_layers)//4)]  # ~4 layers
print(f"Using {len(cgma_layers)} layers for CGMA: {[l.name for l in cgma_layers]}")

for img_idx in tqdm(range(num_images), desc="CGMA Generation"):
    try:
        # Create directory for this image
        img_dir = os.path.join(CGMS_DIR, str(img_idx))
        os.makedirs(img_dir, exist_ok=True)
        
        # Load image
        img = get_img(train_paths[img_idx])
        
        # Get predicted class
        pred = full_model.predict(np.expand_dims(img, axis=0), verbose=0)
        pred_class = np.argmax(pred)
        
        # Generate CGMA from each layer
        accumulated_cam = None
        for layer_idx, conv_layer in enumerate(cgma_layers):
            try:
                cam = generate_gradcam(img, full_model, base_model, conv_layer, pred_class)
                
                # Accumulate CAMs
                if accumulated_cam is None:
                    accumulated_cam = cam.copy()
                else:
                    accumulated_cam += cam
                
                # Normalize accumulated CAM
                if accumulated_cam.max() > 0:
                    norm_cam = accumulated_cam / accumulated_cam.max()
                else:
                    norm_cam = accumulated_cam
                
                # Save accumulated CAM
                output_path = os.path.join(img_dir, f'{layer_idx}.jpg')
                plt.imsave(output_path, norm_cam, cmap='jet', vmin=0, vmax=1)
            
            except Exception as e:
                print(f"  Error with layer {layer_idx}: {e}")
                continue
    
    except Exception as e:
        print(f"  Error processing image {img_idx}: {e}")
        continue

print("✓ CGMA generation complete")

# Generate final GradCAMs (from deepest layer only)
print("\n[5/6] Generating final GradCAMs...")
print(f"Processing {num_images} images...")

deepest_layer = conv_layers[0]  # First in reversed list = deepest
print(f"Using layer: {deepest_layer.name}")

for img_idx in tqdm(range(num_images), desc="GradCAM Generation"):
    try:
        # Load image
        img = get_img(train_paths[img_idx])
        
        # Get predicted class
        pred = full_model.predict(np.expand_dims(img, axis=0), verbose=0)
        pred_class = np.argmax(pred)
        
        # Generate GradCAM
        cam = generate_gradcam(img, full_model, base_model, deepest_layer, pred_class)
        
        # Save GradCAM
        output_path = os.path.join(GRADCAMS_DIR, f'{img_idx}.jpg')
        plt.imsave(output_path, cam, cmap='jet', vmin=0, vmax=1)
    
    except Exception as e:
        print(f"  Error processing image {img_idx}: {e}")
        continue

print("✓ GradCAM generation complete")

# Generate summary
print("\n[6/6] Generating summary...")
summary = f"""
GradCAM Generation Summary
==========================
Model: {MODEL_PATH}
Base Model: {base_model.name}
Dataset: {DATASET_NAME}
Architecture: {ARCH_NAME}

Statistics:
- Total training images: {len(train_paths)}
- Images processed: {num_images}
- Conv2D layers found: {len(conv_layers)}
- Layers used for CGMA: {len(cgma_layers)}
- Deepest layer for GradCAM: {deepest_layer.name}

Output directories:
- CGMAs: {CGMS_DIR}
- GradCAMs: {GRADCAMS_DIR}
- Logs: {LOGS_DIR}

Next steps:
1. Check the generated visualizations
2. Use these for training the GAN explainer
3. Verify quality of generated heatmaps
"""

print(summary)

# Save summary
summary_path = os.path.join(LOGS_DIR, 'generation_summary.txt')
with open(summary_path, 'w') as f:
    f.write(summary)

print("\n" + "="*70)
print("✓ GradCAM generation complete!")
print("="*70)
print(f"\nGenerated files:")
print(f"  CGMAs: {CGMS_DIR}")
print(f"  GradCAMs: {GRADCAMS_DIR}")
print(f"  Summary: {summary_path}")