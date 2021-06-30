import cv2

img = cv2.imread("gamma.jpg",0)
print(img.shape)
img_p = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
cv2.imwrite("img_p.jpg",img_p)