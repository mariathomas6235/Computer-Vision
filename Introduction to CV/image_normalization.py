#This script demonstrates basic image normalization techniques including min-max scaling, mean image creation, and standardization, with visualization of the results.


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to the input grayscale image
image_path = '/content/download.jpeg'

# Read the image in grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Print basic statistics of the image
print(f"Max: {img.max()}, Min: {img.min()}, Mean: {img.mean():.2f}, Std: {img.std():.2f}")

# Min-Max Scaling: scale pixel values to range [0, 255]
minmax = (img - img.min()) / (img.max() - img.min())
img_minmax_scaled = (minmax * 255).astype(np.uint8)

# Mean Image: create an image filled with the mean pixel value of the original image
mean_scaled = np.full_like(img, fill_value=img.mean())
mean_vis = cv2.normalize(mean_scaled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Standardization: zero mean and unit variance, then scale to [0, 255] for visualization
img_std = (img - img.mean()) / img.std()
img_std_scaled = cv2.normalize(img_std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Plot original and processed images side by side
plt.figure(figsize=(10, 10))

plt.subplot(1, 4, 1)
plt.title('Original')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Min-Max Scaled')
plt.imshow(img_minmax_scaled, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Mean Image')
plt.imshow(mean_vis, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Standardized')
plt.imshow(img_std_scaled, cmap='gray')
plt.axis('off')

plt.show()
