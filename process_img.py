import numpy as np
import pandas as pd
import pydicom
import cv2 
import os

def apply_window(hu_image, center, width):
    """
    Apply windowing to HU image and normalize to 0-1.
    """
    lower = center - (width / 2)
    upper = center + (width / 2)

    windowed = np.clip(hu_image, lower, upper)
    normalized = (windowed - lower) / width

    return normalized.astype(np.float32)

import cv2
import numpy as np

def crop_and_letterbox_image(image, target_size=(299, 299), threshold_value=100):
    # Convert 16-bit image to 8-bit grayscale for edge detection
    fake_img = np.mean(image, axis=-1)
    image_8bit = np.uint8((fake_img - np.min(fake_img)) / (np.max(fake_img) - np.min(fake_img)) * 255)

    # Apply threshold to get binary image
    _, binary_image = cv2.threshold(image_8bit, threshold_value, 255, cv2.THRESH_BINARY)

    # Optional morphological operations (helps if noisy)
    kernel = np.ones((5, 5), np.uint8)
    processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Find contours and select largest
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Crop coordinates from the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop original 16-bit image
    cropped_image = image[y:y+h, x:x+w]

    # Letterbox resize (keeping aspect ratio, padding with zeros)
    orig_h, orig_w = cropped_image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w, pad_h = (target_w - new_w) // 2, (target_h - new_h) // 2

    pad_top, pad_bottom = pad_h, target_h - new_h - pad_h
    pad_left, pad_right = pad_w, target_w - new_w - pad_w

    padded_image = cv2.copyMakeBorder(resized_image,
                                      pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=0.0)

    return padded_image


def preprocess_with_standard_windows(dicom_path, metadata_df):
    """
    Reads a DICOM file, applies original metadata-based windowing and two standard windowings (brain and stroke).
    """
    # Standardize paths to avoid mismatches
    dicom_path = os.path.normpath(dicom_path)
    metadata_df['file_path'] = metadata_df['file_path'].apply(os.path.normpath)

    # Ensure exact match with metadata
    metadata_row = metadata_df.loc[metadata_df['file_path'] == dicom_path]

    if metadata_row.empty:
        print(f"Skipping {dicom_path}: No matching metadata found.")
        return None  # Skip processing if no metadata

    # Extract metadata safely
    try:
        rescale_slope = float(metadata_row['RescaleSlope'].values[0])
        rescale_intercept = float(metadata_row['RescaleIntercept'].values[0])
        window_center_meta = float(metadata_row['WindowCenter'].values[0][0])  
        window_width_meta = float(metadata_row['WindowWidth'].values[0][0])  

    except KeyError as e:
        print(f"Metadata error in {dicom_path}: {e}")
        return None  # Skip this file if metadata is missing

    # Read DICOM image
    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array.astype(np.float32)

    # Convert raw pixels to Hounsfield Units
    hu_image = pixel_array * rescale_slope + rescale_intercept

    # Apply windowing
    channel_meta = apply_window(hu_image, window_center_meta, window_width_meta)
    channel_brain = apply_window(hu_image, 40, 80)
    channel_stroke = apply_window(hu_image, 32, 8)

    # Stack channels
    final_image = np.stack([channel_meta, channel_brain, channel_stroke], axis=-1).astype(np.float32)

    return crop_and_letterbox_image(final_image)


'''
def preprocess_with_standard_windows(dicom_path, metadata_df):
    """
    Reads a DICOM file, applies original metadata-based windowing and two standard windowings (brain and stroke),
    and resizes the images preserving diagnostic precision.

    Args:
        dicom_path (str): Path to the DICOM file.
        metadata_df (pd.DataFrame): DataFrame containing metadata.
        target_shape (tuple): Desired shape (height, width).

    Returns:
        np.ndarray: Processed image of shape (512, 512, 3), dtype=float32.
    """

    # Load metadata for given dicom_path
    metadata_row = metadata_df[metadata_df['file_path'] == dicom_path]

    if metadata_row.empty:
        raise ValueError(f"Metadata not found for {dicom_path}")

    # Extract metadata
    rescale_slope = float(metadata_row['RescaleSlope'].values[0])
    rescale_intercept = float(metadata_row['RescaleIntercept'].values[0])

    window_center_meta = float(metadata_row['WindowCenter'].values[0][0])
    window_width_meta = float(metadata_row['WindowWidth'].values[0][0])

    # Standard window settings
    brain_window_center, brain_window_width = 40, 80
    stroke_window_center, stroke_window_width = 32, 8

    # Read DICOM image
    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array.astype(np.float32)

    # Convert raw pixels to Hounsfield Units
    hu_image = pixel_array * rescale_slope + rescale_intercept

    # Channel 1: Original metadata-based windowing
    channel_meta = apply_window(hu_image, window_center_meta, window_width_meta)

    # Channel 2: Brain window
    channel_brain = apply_window(hu_image, brain_window_center, brain_window_width)

    # Channel 3: Stroke window
    channel_stroke = apply_window(hu_image, stroke_window_center, stroke_window_width)

    # Stack channels
    final_image = np.stack([
        channel_meta,
        channel_brain,
        channel_stroke
    ], axis=-1).astype(np.float32)

    padded_img = crop_and_letterbox_image(final_image)

    return padded_img

'''









