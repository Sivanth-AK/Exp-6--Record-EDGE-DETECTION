import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("image.jpg")

# Convert to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ------------------------
# Sobel Edge Detector
# ------------------------
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = np.uint8(sobel)

# ------------------------
# Prewitt Edge Detector
# ------------------------
kernelx = np.array([[1,0,-1],
                    [1,0,-1],
                    [1,0,-1]])

kernely = np.array([[1,1,1],
                    [0,0,0],
                    [-1,-1,-1]])

prewittx = cv2.filter2D(gray, -1, kernelx)
prewitty = cv2.filter2D(gray, -1, kernely)
prewitt = cv2.addWeighted(prewittx, 0.5, prewitty, 0.5, 0)

# ------------------------
# Roberts Edge Detector
# ------------------------
robertsx = np.array([[1,0],
                     [0,-1]])

robertsy = np.array([[0,1],
                     [-1,0]])

robertx = cv2.filter2D(gray, -1, robertsx)
roberty = cv2.filter2D(gray, -1, robertsy)
roberts = cv2.addWeighted(robertx, 0.5, roberty, 0.5, 0)

# ------------------------
# Laplacian Edge Detector
# ------------------------
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# ------------------------
# Canny Edge Detector
# ------------------------
canny = cv2.Canny(gray, 100, 200)

# ------------------------
# Display Images
# ------------------------
plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(sobel, cmap="gray")
plt.title("Sobel")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(prewitt, cmap="gray")
plt.title("Prewitt")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(roberts, cmap="gray")
plt.title("Roberts")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(canny, cmap="gray")
plt.title("Canny")
plt.axis("off")

plt.tight_layout()
plt.show()