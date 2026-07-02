import cv2
import numpy as np

# =====================================================
# IMAGE SETTINGS
# =====================================================

IMAGE_SIZE = (224, 224)

# =====================================================
# LOAD IMAGE
# =====================================================

def load_image(image_path):
    """
    Load image from disk
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


# =====================================================
# PREPROCESS IMAGE
# =====================================================

def preprocess_image(image):
    """
    Resize and normalize image
    """

    image = cv2.resize(image, IMAGE_SIZE)

    image = image.astype(np.float32)

    image = image / 255.0

    return image


# =====================================================
# MODEL INPUT
# =====================================================

def prepare_image(image_path):
    """
    Returns image ready for encoder
    Shape:
    (1,224,224,3)
    """

    image = load_image(image_path)

    image = preprocess_image(image)

    image = np.expand_dims(image, axis=0)

    return image


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    sample = "dataset/processed_images"

    print("Image Utility Ready")