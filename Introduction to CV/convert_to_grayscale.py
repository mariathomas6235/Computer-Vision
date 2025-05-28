"""
Simple script to read an image, convert it to grayscale,
save the grayscale image, and display it using OpenCV

"""

import cv2
import cv2_imshow

# Read the image from a file path
image = cv2.imread('/content/Image.jpeg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Save the grayscale image
cv2.imwrite('gray_image.jpg', gray_image)

# Display the grayscale image
cv2.imshow('Grayscale Image', gray_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
