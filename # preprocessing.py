# preprocessing.py

import kagglehub
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def run_preprocessing(dataset_name="preetviradiya/brian-tumor-dataset", target_size=(224, 224), test_split_ratio=0.2, random_seed=42):
    """
    Performs data preprocessing for the Brain Tumor Dataset.

    Args:
        dataset_name (str): The name of the Kaggle dataset to download.
        target_size (tuple): A tuple (width, height) for resizing images.
        test_split_ratio (float): The proportion of the dataset to include in the test split.
        random_seed (int): Seed for random number generation for reproducibility.

    Returns:
        tuple: X_train, X_test, y_train, y_test (numpy arrays of preprocessed data and labels).
    """

    IMG_WIDTH, IMG_HEIGHT = target_size
    TARGET_SIZE = target_size

    data = []
    labels = []

    print("--- Starting Data Preprocessing ---")

    # 1. Download Dataset
    print(f"Checking for dataset: {dataset_name}")
    # In a typical script, you might define 'path' or check if data is already downloaded
    # For simplicity, this assumes a fresh download or that 'path' might be available
    # if this script is run within an environment where kagglehub was already used.
    # For a standalone script, you'd ensure 'path' is always determined here.
    path = kagglehub.dataset_download(dataset_name)
    print(f"Dataset root path: {path}")

    data_root_path = os.path.join(path, 'Brain Tumor Data Set', 'Brain Tumor Data Set')
    print(f"Data will be loaded from: {data_root_path}")

    class_names = os.listdir(data_root_path)
    print(f"Detected classes: {class_names}")

    print(f"Starting image processing (resizing to {TARGET_SIZE}, normalizing, and labeling)...")

    # 2. Process Images
    for class_name in class_names:
        class_path = os.path.join(data_root_path, class_name)
        if os.path.isdir(class_path):
            # Assign numerical label: 1 for 'Brain Tumor', 0 for 'Healthy'
            label = 1 if class_name == 'Brain Tumor' else 0

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Resize image
                        img = cv2.resize(img, TARGET_SIZE)
                        # Normalize pixel values to [0, 1]
                        img = img.astype('float32') / 255.0

                        data.append(img)
                        labels.append(label)
                except Exception as e:
                    print(f"Could not process image {img_path}: {e}")

    print("Image processing complete.")

    # 3. Convert to Numpy Arrays
    X = np.array(data)
    y = np.array(labels)

    print(f"Shape of preprocessed data (X): {X.shape}")
    print(f"Shape of labels (y): {y.shape}")

    # 4. Split Data into Training and Testing Sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_ratio, random_state=random_seed, stratify=y)

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    print("--- Data Preprocessing Complete ---")

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Example usage when running the script directly
    print("Running preprocessing as a standalone script...")
    X_train_data, X_test_data, y_train_data, y_test_data = run_preprocessing()

    print("Preprocessing finished. You can now use X_train_data, X_test_data, y_train_data, y_test_data")
    