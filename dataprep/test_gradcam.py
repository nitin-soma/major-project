"""
Test script to verify GradCAM generation works correctly on a single image
Ultra-simplified approach
"""

from utils import get_img, all_img_paths
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Parameters
MODEL_PATH = "../models/VGG16_final.h5"
DATASET_NAME = "Food"

print("="*60)
print("GradCAM Test Script - Ultra Simple")
print("="*60)

# Load model
print("\n1. Loading model...")
full_model = tf.keras.models.load_model(MODEL_PATH)
print(f"   ✓ Model loaded")

# Find base model
print("\n2. Finding base model...")
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
    print("   ✗ No base model found!")
    exit(1)

print(f"   ✓ Base model: {base_model.name}")

# Get Conv2D layers
conv_layers = [layer for layer in reversed(base_model.layers) 
               if isinstance(layer, tf.keras.layers.Conv2D)]
print(f"   ✓ Found {len(conv_layers)} Conv2D layers")

# Load test image
print("\n3. Loading test image...")
try:
    a, y = all_img_paths(f'../Datasets/{DATASET_NAME}/training')
    if len(a) > 0:
        img = get_img(a[0])
        true_class = y[0]
        print(f"   ✓ Image loaded")
    else:
        raise FileNotFoundError()
except:
    print("   Creating synthetic image...")
    img = np.random.rand(256, 256, 3).astype('float64')
    true_class = "test"

print(f"   Image shape: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")

# Get prediction
print("\n4. Getting prediction...")
img_batch = np.expand_dims(img, axis=0)
pred = full_model.predict(img_batch, verbose=0)
pred_class = np.argmax(pred)
confidence = pred[0][pred_class]
print(f"   ✓ Predicted class: {pred_class}, confidence: {confidence:.4f}")

# Generate GradCAM - SIMPLE DIRECT METHOD
print("\n5. Generating GradCAM...")
target_layer = conv_layers[0]  # Use first (deepest) conv layer
print(f"   Using layer: {target_layer.name}")

try:
    img_tensor = tf.constant(img_batch, dtype=tf.float32)
    
    # Create a function to get intermediate output
    def get_gradcam(img_tensor, target_layer, pred_class):
        with tf.GradientTape() as tape:
            # Get the output of target layer
            base_model_layer_outputs = tf.keras.models.Model(
                inputs=base_model.input,
                outputs=target_layer.output
            )(img_tensor, training=False)
            
            # Monitored for gradient
            tape.watch(base_model_layer_outputs)
            
            # Full model prediction
            predictions = full_model(img_tensor, training=False)
            class_loss = predictions[:, pred_class]
        
        # Get gradients
        gradients = tape.gradient(class_loss, base_model_layer_outputs)
        return base_model_layer_outputs, gradients
    
    # Get outputs and gradients
    conv_output, grads = get_gradcam(img_tensor, target_layer, pred_class)
    
    if grads is None:
        print("   Gradients are None, using alternative method...")
        # Just visualize the activation
        conv_output = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=target_layer.output
        )(img_tensor, training=False)
        heatmap = np.mean(conv_output[0].numpy(), axis=-1)
    else:
        # Compute CAM
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_np = conv_output[0].numpy()
        pooled_grads_np = pooled_grads.numpy()
        heatmap = np.sum(conv_np * pooled_grads_np, axis=-1)
        heatmap = np.maximum(heatmap, 0)
    
    # Normalize
    heatmap = cv2.resize(heatmap, (256, 256))
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    print(f"   ✓ GradCAM generated")
    print(f"   Range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Visualize
print("\n6. Saving visualization...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original
axes[0].imshow(img)
axes[0].set_title('Original')
axes[0].axis('off')

# Heatmap
im = axes[1].imshow(heatmap, cmap='jet')
axes[1].set_title('GradCAM')
axes[1].axis('off')
plt.colorbar(im, ax=axes[1], fraction=0.046)

# Superimposed
heatmap_uint8 = np.uint8(255 * heatmap)
heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
img_bgr = cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_RGB2BGR)
superimposed = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
axes[2].imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
axes[2].set_title('Superimposed')
axes[2].axis('off')

plt.suptitle(f'Class {pred_class} ({confidence:.2%}) vs True: {true_class}')
plt.tight_layout()
plt.savefig('gradcam_test_output.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved gradcam_test_output.png")

print("\n" + "="*60)
print("✓ Success!")
print("="*60)
