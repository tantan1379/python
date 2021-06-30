from matplotlib.pyplot import *
import numpy as np
from numpy import fft
import math
import cv2


# 估算运动模糊的退化模型
def motion_process(image_size, motion_angle, motion_distance):
    PSF = np.zeros(image_size)
    x_center = (image_size[0] - 1) / 2
    y_center = (image_size[1] - 1) / 2

    sin_val = math.sin(motion_angle * math.pi / 180)
    cos_val = math.cos(motion_angle * math.pi / 180)

    for i in range(motion_distance):
        x_offset = round(i * sin_val)
        y_offset = round(i * cos_val)
        PSF[int(x_center + x_offset), int(y_center - y_offset)] = 1

    return PSF / PSF.sum()


# 对图像进行运动模糊
def motion_blur(img_input, PSF_input, eps):
    img_fft = fft.fft2(img_input)
    PSF_fft = fft.fft2(PSF_input) + eps
    blurred = fft.ifft2(img_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred


#  逆滤波器
def inverse(img_blurred, PSF_input, eps):
    img_blurred_fft = fft.fft2(img_blurred)
    PSF_fft = fft.fft2(PSF_input) + eps
    result = fft.ifft2(img_blurred_fft / PSF_fft)
    result = np.abs(fft.fftshift(result))
    return result


#  维纳滤波器
def wiener(img_input, PSF_input, eps, K=0.01):
    input_fft = fft.fft2(img_input)
    PSF_fft = fft.fft2(PSF_input) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result


# 模拟附加高斯噪声
def noised(pic):
    pic_noised = pic + 0.1 * pic.std() * \
                 np.random.randn(pic.shape[0], pic.shape[1])
    return pic_noised


if __name__ == "__main__":
    img = imread("1.bmp")
    img, g, b ,d= cv2.split(img)
    PSF = motion_process((img.shape[0], img.shape[1]),135,60)
    img_noised = noised(img)
    img_noised_blurred = motion_blur(img_noised, PSF, 0.3)
    img_wiener_deblurred = wiener(img_noised_blurred, PSF, 0.3)
    img_inverse_deblurred = inverse(img_noised_blurred, PSF, 0.3)

    figure()
    gray()
    subplot(231)
    xlabel("origin")
    imshow(img)
    subplot(232)
    xlabel("noised")
    imshow(img_noised)
    subplot(233)
    xlabel("noised_blurred")
    imshow(img_noised_blurred)
    subplot(234)
    xlabel("inverse_deblur")
    imshow(img_inverse_deblurred)
    subplot(235)
    xlabel("img_wiener_deblurred")
    imshow(img_wiener_deblurred)
    show()
