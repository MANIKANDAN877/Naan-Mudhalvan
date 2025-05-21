import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
image = cv2.imread('download (1).jfif')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize the image
resized = cv2.resize(gray, (224, 224))  # Example size

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(resized, (5, 5), 0)

# Thresholding
_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

# Edge detection using Canny

edges = cv2.Canny(blurred, 100, 200)

# Show images
titles = ['Original', 'Gray', 'Resized', 'Blurred', 'Threshold', 'Edges']
images = [image, gray, resized, blurred, thresh, edges]

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

