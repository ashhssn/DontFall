import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = './dataset'
IMG_SIZE = (224, 224)  # size MobileNet uses

def load_data():
    images = []
    labels = []
    for label, folder in enumerate(['no_fall', 'fall']):
        folder_path = os.path.join(DATA_DIR, folder)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

if __name__ == "__main__":
    images, labels = load_data()
    print("Total images:", len(images))
    # Save preprocessed data or split as needed
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    # Save splits as numpy arrays (optional)
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    print("Data prepared and saved.")
