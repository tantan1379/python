#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: main.py
# author: twh
# time: 2020/10/13 14:20

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def read_picture_and_dir():
    imgs, names = [], []
    # 指定图片文件夹
    print('=' * 20 + '选定文件夹' + '=' * 20)
    cwd = os.getcwd()
    path = cwd + "\\picture"
    valid_exts = ['.jpg', 'png', 'jpeg']
    print("%d files in %s" % (len(os.listdir(path)), path))

    # 读取指定文件夹的所有图片，并存放在imgs列表中，保存图片名在names列表中
    print('=' * 20 + '读取所有图像' + '=' * 20)
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_exts:
            print("invalid type")
            continue
        fullpath = os.path.join(path, f)
        imgs.append(cv2.imread(fullpath))
        names.append(f)
    print("%d images loaded" % (len(imgs)))
    return imgs, names, path, cwd


class PictureProcess(object):
    def __init__(self, imgs, names, mag, path, cwd):
        self.imgs = imgs
        self.names = names
        self.mag = mag
        self.path = path
        self.cwd = cwd
        self.N = len(imgs)

    def getall(self):
        return self.imgs, self.names, self.mag, self.path, self.cwd, self.N

    def transform(self):  # 变换 mag为缩放的倍数(mag为正数）
        imgs, names, mag, path, cwd, N = self.getall()
        imgs_gray, imgs_zoom, imgs_map, imgs_back = [], [], [], []

        print('=' * 20 + '对图片进行四种变换' + '=' * 20)
        for i in range(N):
            gray = cv2.imread(path + '\\' + names[i], cv2.IMREAD_GRAYSCALE)
            imgs_gray.append(gray)  # 转换为灰色图像
            width = int(imgs[i].shape[0] * mag)
            height = int(imgs[i].shape[1] * mag)
            dim = (height, width)
            zoom = cv2.resize(imgs[i], dim)
            imgs_zoom.append(zoom)
            map = np.array(imgs[i]) / 255
            imgs_map.append(map)  # 映射到0-1
            back = 255 - imgs[i]
            imgs_back.append(back)  # 反向处理
        # 显示并保存变换后的图片
        for i in range(N):
            cv2.imshow('imgs', imgs[i])
            cv2.imwrite(r"./processing/imgs%d.jpg" % i, imgs[i])
            cv2.imshow('imgs_gray', imgs_gray[i])
            cv2.imwrite(r"./processing/imgs_gray%d.jpg" % i, imgs_gray[i])
            cv2.imshow('imgs_zoom', imgs_zoom[i])
            cv2.imwrite(r"./processing/imgs_zoom%d.jpg" % i, imgs_zoom[i])
            cv2.imshow('imgs_map', imgs_map[i])
            cv2.imwrite(r"./processing/imgs_map%d.jpg" % i, imgs_map[i] * 255)
            cv2.imshow('imgs_back', imgs_back[i])
            cv2.imwrite(r"./processing/imgs_back%d.jpg" % i, imgs_back[i])
            cv2.waitKey(2000)  # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
            cv2.destroyAllWindows()
        print("图形变换已完成")

    def compute_contour_histogram(self):
        # 计算轮廓和直方图
        imgs, names, mag, path, cwd, N = self.getall()
        imgs_gray = []
        print('=' * 20 + '计算轮廓和直方图' + '=' * 20)
        for i in range(N):
            gray = cv2.imread(path + '\\' + names[i], cv2.IMREAD_GRAYSCALE)
            imgs_gray.append(gray)
            plt.figure()
            plt.gray()
            plt.subplot(221)
            plt.contour(imgs_gray[i], origin='image')  # 计算轮廓
            plt.ion()
            plt.subplot(222)
            plt.imshow(imgs_gray[i])
            plt.subplot(223)
            plt.hist(imgs_gray[i].flatten(), 256)  # 计算直方图
            plt.savefig(r"./processing\img_out_outline&histogram%d.jpg" % i)
            plt.ion()
        plt.pause(1.5)  # 显示秒数
        plt.close('all')
        print("轮廓及直方图计算完成")

    def histogram_equalization(self):
        # 直方图均衡化
        imgs, names, mag, path, cwd, N = self.getall()
        frameH = []
        imgs_1 = [0] * len(imgs)
        print('=' * 20 + '直方图均衡化' + '=' * 20)
        for i in range(len(imgs)):
            (b, g, r) = cv2.split(imgs[i])
            bH = cv2.equalizeHist(b)
            gH = cv2.equalizeHist(g)
            rH = cv2.equalizeHist(r)
            frameH.append(cv2.merge((bH, gH, rH)))
            plt.figure()
            plt.subplot(221)
            imgs_1[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
            frameH[i] = cv2.cvtColor(frameH[i], cv2.COLOR_BGR2RGB)
            plt.imshow(imgs_1[i])
            plt.subplot(222)
            plt.hist(imgs[i].flatten(), 128)
            plt.ylim((0, 35000))
            plt.subplot(223)
            plt.imshow(frameH[i])
            plt.subplot(224)
            plt.hist(frameH[i].flatten(), 256)
            plt.ylim((0, 35000))
            plt.ion()
            plt.savefig(r"./processing/img_histogram_equalization%d.jpg" % i)
        plt.pause(1.5)
        plt.close('all')
        print("直方图均衡化已完成")

    def picture_filter(self):
        # 图像滤波
        imgs, names, mag, path, cwd, N = self.getall()
        result_1, result_2, result_3 = [], [], []
        result_0=imgs
        print('=' * 20 + '图像滤波' + '=' * 20)

        for i in range(len(imgs)):
            result_0[i] = cv2.cvtColor(result_0[i], cv2.COLOR_BGR2RGB)
            result_1.append(cv2.blur(result_0[i], (8, 8)))
            result_2.append(cv2.GaussianBlur(result_0[i], (5, 5), 0))
            result_3.append(cv2.medianBlur(result_0[i], 5))
            plt.figure()
            plt.subplot(232)
            plt.imshow(result_0[i])
            plt.title('origin')
            plt.subplot(234)
            plt.imshow(result_1[i])
            plt.title('mean')
            plt.subplot(235)
            plt.imshow(result_2[i])
            plt.title('gaussian')
            plt.subplot(236)
            plt.imshow(result_3[i])
            plt.title('median')
            plt.ion()
            plt.savefig(r"./processing\img_average_filtering%d.jpg" % i)
        plt.pause(3)
        plt.close('all')
        print("图像滤波已完成")


def main():
    imgs, names, path, cwd = read_picture_and_dir()
    P = PictureProcess(imgs, names, 2, path, cwd)
    P.transform()
    P.compute_contour_histogram()
    P.histogram_equalization()
    P.picture_filter()


if __name__ == '__main__':
    main()
