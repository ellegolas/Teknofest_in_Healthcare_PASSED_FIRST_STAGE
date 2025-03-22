import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pydicom
import numpy as np
from sklearn.model_selection import train_test_split
from process_img import preprocess_with_standard_windows


def read_image_df(data_dir):
    """Load image paths and labels from Google Drive dataset."""
    data = []
    
    for label in ["0", "1"]:  # Assuming the dataset is organized in 0/ and 1/ subfolders
        label_path = os.path.join(data_dir, label)
        
        if not os.path.exists(label_path):
            print(f"Warning: Directory {label_path} not found!")
            continue
        
        for file in os.listdir(label_path):
            if file.lower().endswith('.dcm'):  # Ensure only DICOM files are read
                file_path = os.path.join(label_path, file)
                data.append({"file_path": file_path, "label": int(label)})
    
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No images found! Check Google Drive path.")
    else:
        print(f"Loaded {len(df)} images from {data_dir}")
    
    return df

def process_window_value(window_value):
    if isinstance(window_value, pydicom.multival.MultiValue):
        return list(window_value) 
    else:
        return [float(window_value)]

def read_metadata_df(data_dir):
    """Load metadata from DICOM files into a DataFrame."""
    metadata_list = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.dcm'):
                file_path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(file_path)
                    metadata_list.append({
                        "file_path": file_path,
                        "SliceThickness": float(ds.SliceThickness),
                        "RescaleSlope": float(ds.RescaleSlope),
                        "RescaleIntercept": float(ds.RescaleIntercept),
                        "WindowCenter": [float(x) for x in ds.WindowCenter] if isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else [float(ds.WindowCenter)],
                        "WindowWidth": [float(x) for x in ds.WindowWidth] if isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else [float(ds.WindowWidth)],
                        "PixelSpacing": process_window_value(ds.PixelSpacing),
                    })
                except Exception as e:
                    print(f"Error reading metadata from {file_path}: {e}")
    return pd.DataFrame(metadata_list)


def split_dataset(df):
    """Split dataset into train (70%), validation (20%), test (10%)."""
    train_val, test = train_test_split(df, test_size=0.10, stratify=df['label'], random_state=42)
    train, val = train_test_split(train_val, test_size=0.2222, stratify=train_val['label'], random_state=42)
    return train, val, test


def load_and_process_images(df, metadata_df):
    """Load and process images, ensuring alignment with metadata."""
    processed_images = []
    slice_thickness_list = []
    pixel_spacing_first_elem_list = []

    for _, row in df.iterrows():
        try:
            processed_image = preprocess_with_standard_windows(row['file_path'], metadata_df)
            processed_images.append(processed_image)

            # Fetch metadata row
            metadata_row = metadata_df[metadata_df['file_path'] == row['file_path']]

            # SliceThickness
            slice_thickness = metadata_row['SliceThickness'].values
            slice_thickness_list.append(slice_thickness[0] if len(slice_thickness) > 0 else 0)

            # PixelSpacing[0] only
            pixel_spacing = metadata_row['PixelSpacing'].values
            if len(pixel_spacing) > 0:
                spacing_val = pixel_spacing[0]
                if isinstance(spacing_val, str) and '[' in spacing_val:
                    # Convert from string to list
                    spacing_list = [float(v.strip()) for v in spacing_val.strip('[]').split(',')]
                    pixel_spacing_first_elem_list.append(spacing_list[0])
                elif isinstance(spacing_val, (list, np.ndarray)):
                    pixel_spacing_first_elem_list.append(spacing_val[0])
                else:
                    pixel_spacing_first_elem_list.append(float(spacing_val))
            else:
                pixel_spacing_first_elem_list.append(0.0)

        except Exception as e:
            print(f"Error processing {row['file_path']}: {e}")
            slice_thickness_list.append(0)
            pixel_spacing_first_elem_list.append(0.0)

    processed_images = np.array(processed_images)
    slice_thickness_array = np.array(slice_thickness_list).reshape(-1, 1)
    pixel_spacing_array = np.array(pixel_spacing_first_elem_list).reshape(-1, 1)

    # Combine the two features
    metadata_features = np.hstack((slice_thickness_array, pixel_spacing_array))

    # Normalize to 0–1
    scaler = MinMaxScaler()
    normalized_metadata = scaler.fit_transform(metadata_features)

    if processed_images.shape[0] != normalized_metadata.shape[0]:
        raise ValueError("Mismatch between processed images and metadata features. Check file paths and metadata alignment.")

    return processed_images, normalized_metadata



