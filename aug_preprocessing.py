import os
import cv2
import shutil
import numpy as np
import imgaug.augmenters as iaa

# Define the folders for cancer and non-cancer images
cancer_folder = 'E:\CS_354_final_2\classified_images\images_malignant'
non_cancer_folder = 'E:\CS_354_final_2\classified_images\images_benign'

# Define the target size for images after preprocessing
target_size = (300, 300)

# Create output folders if they don't exist
os.makedirs(cancer_folder, exist_ok=True)
os.makedirs(non_cancer_folder, exist_ok=True)

# Define augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scaling images
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # translating images
        rotate=(-45, 45),  # rotating images
        shear=(-16, 16),  # shearing images
    ),
    iaa.GaussianBlur(sigma=(0, 1.0)),  # blurring images
    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # adding Gaussian noise
])

def preprocess_and_augment_images(src_folder, dst_folder):
    # Get a list of image files
    image_files = os.listdir(src_folder)
    for image_file in image_files:
        # Read the image
        image_path = os.path.join(src_folder, image_file)
        image = cv2.imread(image_path)

        # Resize the image to target size
        image = cv2.resize(image, target_size)

        # Apply data augmentation
        augmented_images = [image]
        for _ in range(4):  # Augment each image 4 times
            augmented_image = seq.augment_image(image)
            augmented_images.append(augmented_image)

        # Save augmented images directly into the destination folder
        for i, augmented_image in enumerate(augmented_images):
            output_path = os.path.join(dst_folder, f"{os.path.splitext(image_file)[0]}_{i}.jpg")
            cv2.imwrite(output_path, augmented_image)

# Perform preprocessing and augmentation for cancer images
preprocess_and_augment_images(cancer_folder, cancer_folder)

# Perform preprocessing and augmentation for non-cancer images
preprocess_and_augment_images(non_cancer_folder, non_cancer_folder)

print("Data augmentation and preprocessing completed successfully.")
