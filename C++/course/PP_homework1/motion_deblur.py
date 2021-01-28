#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: motion_deblur
# author: twh
# time: 2020/10/22 14:26


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


def motion_blur(img, PSF, eps):
    img_fft = fft.fft2(img)
    PSF_fft = fft.fft2(PSF) + eps
    blurred = fft.ifft2(img_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred


def inverse(img_blurred, PSF, eps):
    img_blurred_fft = fft.fft2(img_blurred)
    PSF_fft = fft.fft2(PSF) + eps
    result = fft.ifft2(img_blurred_fft / PSF_fft)
    result = np.abs(fft.fftshift(result))
    return result


def wiener(input, PSF, eps, K=0.01):
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result


if __name__ == "__main__":
    img = imread("1.bmp")
    img, g, b, d = cv2.split(img)
    PSF = motion_process((img.shape[0], img.shape[1]), 135, 50)
    img_blurred = motion_blur(img, PSF, 1e-3)
    img_restored_inverse = inverse(img_blurred, PSF, 1e-3)
    img_restored_wiener = wiener(img_blurred, PSF, 1e-3)

    # 画图
    figure()
    gray()
    subplot(141)
    xlabel("origin image")
    imshow(img)
    subplot(142)
    xlabel("blurred image")
    imshow(img_blurred)
    subplot(143)
    xlabel("restored image(inverse)")
    imshow(img_restored_inverse)
    subplot(144)
    xlabel("restored image(wiener)")
    imshow(img_restored_wiener)
    show()
