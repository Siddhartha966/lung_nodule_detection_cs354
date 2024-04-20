import json
import os
import shutil

# Path to your annotations file
annotations_file = 'E:\CS_354_final_2\data_train\_annotations.coco.json'

# Path to your images directory
images_dir = 'E:\CS_354_final_2\data_train'

# Create folders to store images with cancer and without cancer
cancer_folder = 'E:\CS_354_final_2\classified_images\images_malignant'
non_cancer_folder = 'E:\CS_354_final_2\classified_images\images_benign'

os.makedirs(cancer_folder, exist_ok=True)
os.makedirs(non_cancer_folder, exist_ok=True)

# Load annotations file
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

# Iterate through annotations and move images to appropriate folders
for annotation in annotations['annotations']:
    image_id = annotation['image_id']
    category_id = annotation['category_id']
    file_name = annotations['images'][image_id]['file_name']

    # Check if category_id corresponds to cancer
    if category_id in [0, 1, 2]:  # Assuming category_ids for cancer-related categories are 0, 1, 2
        src_path = os.path.join(images_dir, file_name)
        dst_path = os.path.join(cancer_folder, file_name)
        shutil.copy(src_path, dst_path)
    else:
        src_path = os.path.join(images_dir, file_name)
        dst_path = os.path.join(non_cancer_folder, file_name)
        shutil.copy(src_path, dst_path)

print("Images have been moved successfully.")
