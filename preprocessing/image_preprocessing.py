import os
import cv2
import numpy as np

# =====================================================
# PATHS
# =====================================================

INPUT_FOLDER = "dataset/image_dataset"
OUTPUT_FOLDER = "dataset/processed_images"

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Supported image extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Resize size
IMAGE_SIZE = (224, 224)

processed = 0
skipped = 0

print("=" * 50)
print("Image Preprocessing Started...")
print("=" * 50)

# =====================================================
# PROCESS IMAGES
# =====================================================

for root, dirs, files in os.walk(INPUT_FOLDER):

    for file in files:

        if file.lower().endswith(IMAGE_EXTENSIONS):

            image_path = os.path.join(root, file)

            image = cv2.imread(image_path)

            if image is None:
                skipped += 1
                continue

            # Resize
            image = cv2.resize(image, IMAGE_SIZE)

            # Normalize
            image = image.astype(np.float32) / 255.0

            # Convert back for saving
            image = (image * 255).astype(np.uint8)

            save_path = os.path.join(OUTPUT_FOLDER, file)

            cv2.imwrite(save_path, image)

            processed += 1

print("\nImage Preprocessing Completed")
print(f"Processed Images : {processed}")
print(f"Skipped Images   : {skipped}")
print("=" * 50)