import numpy as np
import os
import glob
import cv2
import tensorflow as tf

def load_all_data():
    np.random.seed(7)
    files = glob.glob('../npys/*.npy')
    print(files)
    data=np.concatenate([np.load(i,allow_pickle=True)[:5000]for i in files])
    np.random.shuffle(data)
    l= data.shape[0]
    split=int(np.floor(0.1*l))
    test  = data[:split]
    train = data[split:]
    return (train,test)

def process_dataset(data):
    inp_images    = np.array(data[:,0].tolist())/255.
    # Load CNN features instead of CGMs
    features = []
    count = 0
    for i in np.array(data[:,1].tolist()):
        # Assume i is the path or the feature array
        # For now, assume data[:,1] is feature arrays
        features.append(i)
        print(count, end='\r')
        count += 1
    features = np.array(features)
    # Resize features to (64,64,8) if necessary
    # Features are loaded as (1, h, w, c), resize to (64,64,8)
    resized_features = []
    for feat in features:
        feat_resized = tf.image.resize(feat, (64, 64)).numpy()
        # Squeeze batch dimension
        feat_resized = feat_resized[0]  # Now (64,64,c)
        if feat_resized.shape[-1] > 8:
            feat_resized = feat_resized[:, :, :8]
        elif feat_resized.shape[-1] < 8:
            # Pad with zeros
            pad = np.zeros((64, 64, 8 - feat_resized.shape[-1]))
            feat_resized = np.concatenate([feat_resized, pad], axis=-1)
        resized_features.append(feat_resized)
    features = np.array(resized_features)
    out_images    = np.array(data[:,2].tolist())/255.
    y             = np.array(data[:,3].tolist())
    print('features shape = ', features.shape)
    return (inp_images, features, out_images, y)
