"""
src/preprocessing/split_dataset.py
===================================
Reproducible stratified dataset splitting module for PlantVillage dataset.
Splits data into 70% Train, 15% Validation, and 15% Test using random_state=123.
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def find_dataset_root():
    """Finds PlantVillage dataset directory downloaded by kagglehub or present locally."""
    possible_paths = [
        os.path.expanduser("~/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3/plantvillage dataset/color"),
        os.path.expanduser("~/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3/color"),
        os.path.expanduser("~/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/1/plantvillage dataset/color"),
        os.path.expanduser("~/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/1/color"),
        "data/plantvillage/color",
        "plantvillage/color",
        "/kaggle/input/plantvillage-dataset/color"
    ]
    for path in possible_paths:
        if os.path.exists(path) and len(os.listdir(path)) >= 30:
            return path
            
    # Search in user home directory for kagglehub cache
    cache_base = os.path.expanduser("~/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset")
    if os.path.exists(cache_base):
        for root, dirs, files in os.walk(cache_base):
            if "Apple___Apple_scab" in dirs:
                return root
                
    raise FileNotFoundError("PlantVillage dataset directory not found. Run kagglehub download first.")

def create_dataset_splits(seed=123, output_dir="data_splits"):
    data_dir = find_dataset_root()
    print(f"[Dataset Splitter] Found dataset at: {data_dir}")

    available_subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    file_paths = []
    labels = []
    class_map = {}

    for idx, class_name in enumerate(CLASSES):
        class_map[class_name] = idx
        # Find directory matching class_name
        matched_dir = None
        for subdir in available_subdirs:
            if subdir == class_name or subdir.replace("___", " ").replace("_", " ") == class_name.replace("___", " ").replace("_", " "):
                matched_dir = subdir
                break
        if matched_dir is None:
            # Fallback search
            for subdir in available_subdirs:
                if class_name.split("___")[-1].lower() in subdir.lower():
                    matched_dir = subdir
                    break

        if matched_dir is None:
            print(f"[Warning] Subdirectory for class {class_name} not found!")
            continue

        cls_dir = os.path.join(data_dir, matched_dir)
        img_files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for path in img_files:
            file_paths.append(path)
            labels.append(idx)

    file_paths = np.array(file_paths)
    labels = np.array(labels)

    print(f"[Dataset Splitter] Total images found across 38 classes: {len(file_paths)}")

    # 70% Train, 30% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        file_paths, labels, test_size=0.30, random_state=seed, stratify=labels
    )

    # 15% Val, 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    print(f"[Dataset Splitter] Train count: {len(X_train)} (70%)")
    print(f"[Dataset Splitter] Val count:   {len(X_val)} (15%)")
    print(f"[Dataset Splitter] Test count:  {len(X_test)} (15%)")

    os.makedirs(output_dir, exist_ok=True)
    
    splits = {
        "train": [{"path": p, "label": int(l)} for p, l in zip(X_train, y_train)],
        "val": [{"path": p, "label": int(l)} for p, l in zip(X_val, y_val)],
        "test": [{"path": p, "label": int(l)} for p, l in zip(X_test, y_test)]
    }

    for name, data in splits.items():
        split_path = os.path.join(output_dir, f"{name}_split.json")
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[Dataset Splitter] Saved {split_path}")

    return splits

if __name__ == "__main__":
    create_dataset_splits()
