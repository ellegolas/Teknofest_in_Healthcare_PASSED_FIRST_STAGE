import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

from model import get_lr_scheduler, LearningRateLogger
import numpy as np

def augment_and_expand_dataset(images, metadata, labels, n_augmented=1):
    """Generate n_augmented versions per original image, and return expanded arrays."""
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.02,
        height_shift_range=0.02,
        shear_range=2.0,
        fill_mode='nearest'
    )

    augmented_images = []
    augmented_metadata = []
    augmented_labels = []

    for idx in range(len(images)):
        image = images[idx]
        meta = metadata[idx]
        label = labels[idx]

        image = np.expand_dims(image, 0)  # Add batch dimension
        label = np.expand_dims(label, 0)

        aug_iter = datagen.flow(image, label, batch_size=1)

        for _ in range(n_augmented):
            aug_img, _ = next(aug_iter)
            augmented_images.append(aug_img[0])  # remove batch dim
            augmented_metadata.append(meta)
            augmented_labels.append(label[0])

    # Combine original and augmented
    combined_images = np.concatenate([images, np.array(augmented_images)], axis=0)
    combined_metadata = np.concatenate([metadata, np.array(augmented_metadata)], axis=0)
    combined_labels = np.concatenate([labels, np.array(augmented_labels)], axis=0)

    return combined_images, combined_metadata, combined_labels


def train_model(model, train_images, train_metadata, train_labels, val_images, val_metadata, val_labels, batch_size=16, epochs=20):
    """Train model using manually augmented dataset."""
    
    # Apply manual augmentation
    train_images_aug, train_metadata_aug, train_labels_aug = augment_and_expand_dataset(
        train_images, train_metadata, train_labels, n_augmented=2
    )

    print("Train image count before augmentation:", train_images.shape[0])
    print("Train image count after augmentation:", train_images_aug.shape[0])

    # Callbacks
    lr_scheduler = get_lr_scheduler()
    lr_logger = LearningRateLogger()
    early_stopper = EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True,
        verbose=1
    )

    # Train model
    history = model.fit(
        [train_images_aug, train_metadata_aug], train_labels_aug,
        validation_data=([val_images, val_metadata], val_labels),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[lr_scheduler, lr_logger, early_stopper],
        verbose=1
    )

    return history





'''
def train_model(model, train_images, train_metadata, train_labels, val_images, val_metadata, val_labels, batch_size=16, epochs=20):
    """Train model with data augmentation."""
    datagen = ImageDataGenerator(
        rotation_range=5,            # Small safe rotations
        width_shift_range=0.02,
        height_shift_range=0.02,
        shear_range=2.0,             # Mild shearing
        fill_mode='nearest'
    )
    
    datagen.fit(train_images)

    lr_scheduler = get_lr_scheduler()
    lr_logger = LearningRateLogger()  # New callback  

    early_stopper = EarlyStopping(
        monitor='val_loss',
        patience=12,                 # Stop if no improvement for 8 epochs
        restore_best_weights=True, # Roll back to best model
        verbose=1
    )  

    history = model.fit(   
    [train_images, train_metadata], train_labels,
    validation_data=([val_images, val_metadata], val_labels),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[lr_scheduler, lr_logger, early_stopper]  # Include both LR scheduler and logger
    )

    return history
'''

