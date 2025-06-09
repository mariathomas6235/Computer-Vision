import numpy as np

# Define a simple 5x5x3 RGB image (3 channels)
image = np.array([
    [[1, 0, 2], [2, 1, 1], [3, 2, 0], [0, 1, 1], [1, 0, 2]],
    [[0, 1, 0], [1, 0, 1], [2, 2, 2], [3, 1, 3], [2, 0, 1]],
    [[3, 0, 2], [0, 1, 0], [1, 0, 1], [2, 2, 2], [1, 0, 0]],
    [[2, 1, 1], [1, 0, 2], [3, 3, 1], [0, 1, 0], [0, 2, 1]],
    [[1, 2, 2], [2, 1, 0], [0, 0, 1], [1, 2, 2], [2, 1, 1]]
])

# Define a 3x3x3 filter (kernel) for each channel (RGB)
kernel = np.array([
    [[0, 1, 0], [1, -1, 1], [0, 1, 0]],
    [[1, 0, 1], [0, -1, 0], [1, 0, 1]],
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
])

# Convolution operation
def convolve_rgb(image, kernel):
    image_h, image_w, image_c = image.shape
    kernel_h, kernel_w, kernel_c = kernel.shape
    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1
    output = np.zeros((output_h, output_w, 1))

    for k in range(image_c):  # Apply the convolution for each channel
        for i in range(output_h):
            for j in range(output_w):
                output[i, j] = np.sum(image[i:i+kernel_h, j:j+kernel_w, k] * kernel)
    return output

# Apply the convolution
output = convolve_rgb(image, kernel)
print("Convolution Output:\n", output)