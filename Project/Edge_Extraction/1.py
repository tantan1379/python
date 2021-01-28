import cv2
import numpy as np
import matplotlib.pyplot as plt


def imreadex(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)


image = imreadex(r"1.jpg")
img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# cv2.imshow('img', img)
oldimg = img

# sobel
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# cv2.imshow('sobel', dst)

# canny
img = cv2.GaussianBlur(img, (3, 3), 0)
canny = cv2.Canny(img, 100, 200)
# cv2.imshow("canny", canny)

# robert
kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
kernely = np.array([[0, -1], [1, 0]], dtype=int)
x = cv2.filter2D(oldimg, cv2.CV_16S, kernelx)
y = cv2.filter2D(oldimg, cv2.CV_16S, kernely)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# cv2.imshow('robert', Roberts)
# cv2.waitKey(0)

# figure
plt.figure()
plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title('origin')
plt.subplot(1, 4, 2)
plt.imshow(dst, cmap='gray')
plt.title('sobel')
plt.subplot(1, 4, 3)
plt.imshow(canny, cmap='gray')
plt.title('canny')
plt.subplot(1, 4, 4)
plt.imshow(Roberts, cmap='gray')
plt.title('robert')
plt.show()
