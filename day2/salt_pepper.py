import cv2
import numpy as np
import random

def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    Add salt and pepper noise to an image.
    
    Args:
        image: Input image (NumPy array).
        salt_prob: Probability of salt noise (white pixels).
        pepper_prob: Probability of pepper noise (black pixels).
    
    Returns:
        Noisy image (NumPy array).
    """
    noisy_img = np.copy(image)
    total_pixels = image.size

    # Salt noise
    num_salt = np.ceil(salt_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    noisy_img[coords[0], coords[1]] = 255 if len(image.shape) == 2 else [255]*image.shape[2]

    # Pepper noise
    num_pepper = np.ceil(pepper_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    noisy_img[coords[0], coords[1]] = 0 if len(image.shape) == 2 else [0]*image.shape[2]

    return noisy_img

# Example usage
image = cv2.imread("images/dog.png")  # Replace with your image path
noisy_image = add_salt_and_pepper_noise(image, 0.02, 0.02)

cv2.imwrite("images/dog_salt_pepper.png", noisy_image)

