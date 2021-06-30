from skimage import feature, exposure
import cv2
import numpy as np

def addblack(pic,width=10):
    black = np.full((pic[0].shape[0],width,3),200)
    for i,p in enumerate(pic):
        if i==0:
            res = p
            continue
        res = np.hstack([res,black,p])
    return res

def extract(img, point_A, point_B):
    
       h1, w1 = point_A[0], point_A[1]
       h2, w2 = point_B[0], point_B[1]
       return img[h1:h2, w1:w2]

img = cv2.imread('img_p.jpg')
img = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
# img = np.float32(img) / 255.0
# img2 = np.power(img/float(np.max(img)), 1.5)
fd, hog_image = feature.hog(img, orientations=9, pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2), visualize=True)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
print(fd.shape)
# hog_image_rescaled = np.expand_dims(hog_image_rescaled, 2)
# hog_image_rescaled = np.concatenate((hog_image_rescaled, hog_image_rescaled, hog_image_rescaled), axis=-1)
# # res = addblack([img,hog_image_rescaled])
# cv2.imshow('hog_image', hog_image)
# cv2.imshow('img', img)
# cv2.imshow('hog', hog_image_rescaled)
# cv2.imshow('res', res)
# cv2.waitKey(0)


# 计算x和y方向的梯度
# gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
# gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
# # 计算合梯度的幅值和方向（角度）
# mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
# res = addblack([gx,gy,mag,angle])
# cv2.imshow('res',res)
# cv2.waitKey()
# cv2.destroyWindows()