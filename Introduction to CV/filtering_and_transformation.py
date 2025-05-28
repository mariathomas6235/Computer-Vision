import numpy as np
import cv2
import matplotlib.pyplot as plt

# Path to the input image
image_path = '/content/download.jpeg'

def filter_transform(image_path, blur_kernel_size=(5, 5), rotation_angle=45):
    """
    Applies Gaussian blur and rotation transformation to a grayscale image,
    then displays the original and transformed images side by side.

    Args:
        image_path (str): Path to the input image file.
        blur_kernel_size (tuple): Kernel size for Gaussian blur (default (5,5)).
        rotation_angle (float): Angle in degrees to rotate the image (default 45).
    """
    # Read image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, blur_kernel_size, 0)
    
    # Get image dimensions and center
    height, width = blurred.shape
    center = (width // 2, height // 2)
    
    # Calculate rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    
    # Perform affine warp (rotation)
    rotated_img = cv2.warpAffine(blurred, rotation_matrix, (width, height))
    
    # Plot original and transformed images
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Transformed')
    plt.imshow(rotated_img, cmap='gray')
    plt.axis('off')
    
    plt.show()

# call the function
filter_transform(image_path, blur_kernel_size=(7, 7), rotation_angle=30)
