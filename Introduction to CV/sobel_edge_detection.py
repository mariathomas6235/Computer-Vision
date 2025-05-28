"""
Sobel Edge Detection Demo

This script demonstrates:
1. Manual implementation of the Sobel vertical edge detection filter
2. Applying Sobel vertical filter using OpenCV's filter2D function

It reads a grayscale image, applies both methods, and displays the results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to input image - replace with your local image path
img_path = '/content/download.jpeg'

# Read image in grayscale mode
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Define Sobel vertical kernel
sobel_vertical = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Get image dimensions
rows, cols = image.shape

# Kernel size
k = sobel_vertical.shape[0]

# Initialize output matrix for manual convolution result
output_matrix = np.zeros((rows - k + 1, cols - k + 1))

# Manual convolution with Sobel kernel (vertical edges)
for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        region = image[i-1:i+2, j-1:j+2]
        output_matrix[i-1, j-1] = np.sum(region * sobel_vertical)

# Plot original image and manual Sobel output
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Sobel Conv Edge Output (Manual)")
plt.imshow(output_matrix, cmap='gray')
plt.axis('off')

plt.show()


# Apply Sobel vertical filter using OpenCV filter2D
sobel_output = cv2.filter2D(image, -1, sobel_vertical)

# Plot original image and OpenCV filter2D output
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Sobel Edge Output (filter2D)")
plt.imshow(sobel_output, cmap='gray')
plt.axis('off')

plt.show()
