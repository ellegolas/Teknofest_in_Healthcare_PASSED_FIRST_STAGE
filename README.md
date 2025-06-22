# Teknofest Healthcare Artificial Intelligence Competition, Stroke Detection Task from MR images
#### ''' This repo is the repo that my teammates and I use, where we share codes and follow the development of the project. '''

## Dataset and Preprocessing
This project uses brain CT scans from the 2021 Teknofest AI in Healthcare Competition to detect stroke conditions. The dataset consists of:
- 6,650 DICOM CT scans (with metadata: SliceThickness, PixelSpacing)
- Train set: Augmented to 11,000+ images
- Validation set: 1,413 images
- Test set: 998 images

* Due to lack of patient-wise identifiers, we used random train_test_split. To assess generalizability, external validation was performed using RSNA’s public dataset (2,000 images), preprocessed identically. *

## Model Architecture
We employed a fine-tuned InceptionV3 (ImageNet weights) for binary classification:
- Loss: binary_crossentropy
- Optimizer: Adam
- Metrics: Accuracy, Precision, Recall, F1-Score

### Test Set Performance on the best training:
- Accuracy: 0.9759
- F1-Score: 0.9629
- Precision: 0.9904
- Recall: 0.9369

### External Validation (RSNA):
* We applied external validation on RSNA dataset and performance was down drastically becuase of the domain difference. EVEN MR machine model can affect how CNN could behave in such scenario. *
- Accuracy: 0.7160
- F1-Score: 0.6474
- Precision: 0.8595
- Recall: 0.5192
- AUC: 0.7632

* Performance drop on RSNA data highlights domain shift issues caused by differences in scanners, HU value ranges, noise levels, resolution, and labeling techniques. *

## Best and Last Training Process Details:
- Trained for 75 epochs, stopped at epoch 68 using EarlyStopping
- Best validation accuracy at epoch 56 (weights restored)
- Learning rate decayed from 1e-5 to 1e-8 progressively
- Overfitting mitigated with L2 regularization, dropout, and HU-preserving augmentation
- Training conducted on Google Colab Pro due to local GPU limitations, we purchased Colab Pro to train from scratch

## Key Innovations & Original Contributions 

### 1. HU-Based Multi-Window Input
Each image was constructed from 3 different windows, preserving clinical relevance:
- Metadata-derived dynamic window
- Fixed brain window: center=40, width=80
- Fixed stroke window: center=32, width=8

* This allowed the model to learn from clinically meaningful contrasts in HU space. *

### 2. Image Cropping & Letterbox Resize
To eliminate irrelevant brain scan areas:
- Thresholding + morphological operations + contour detection
- Cropping based on bounding box of the largest contour
- Bicubic interpolation used for resizing
- Aspect ratio preserved using letterbox padding to 299x299 !!!

### 3. Multi-Input Model Architecture
Alongside images, we injected DICOM metadata into the model:
- SliceThickness and PixelSpacing were normalized and added as scalar inputs
- Metadata was concatenated with visual features before the final layers

* This enabled hybrid reasoning using both structural and visual information. *

### 4. Medical-Grade Augmentation
HU-preserving augmentation strategies:
- Rotation: ±5°
- Horizontal/vertical shift: max 2%
- Minor shearing
- fill_mode='nearest' to maintain spatial consistency

* No conversion to PNG — augmentation was applied directly to the original DICOM data to preserve spatial information that is hidden in the richness of DICOM images!! *

### 5. Controlled and Stable Learning
- EarlyStopping halted training at optimal epoch and restored best weights
- ReduceLROnPlateau adjusted the learning rate based on validation loss
- Fast convergence was prevented, ensuring robust generalization



### To run in COLAB;

* Open main.ipynb as notebook
* Load other files (ggl_data_loader instead of dataloader, ggl_save_info instead of save_info)
* Connect to your drive and specify the image directory path in the code
* Download requirements.txt with pip install
* When you run the block, main.ipynb will be operational using all other functions
