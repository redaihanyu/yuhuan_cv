# -*- coding: utf-8 -*-
'''
@version: 0.0.1
@author: yuhh
@Contact: redaihanyu@126.com
@site: 
@file: Retinex_self.py
@time: 2018/12/30 11:20 PM

'''
import sys

reload(sys)
sys.setdefaultencoding('utf8')

import cv2
import numpy as np


# 恢复到0-255阈值
def restore(img):
    for i in range(img.shape[2]):
        img_i_min = np.min(img[:, :, i])
        img_i_max = np.max(img[:, :, i])
        img[:, :, i] = (img[:, :, i] - img_i_min) / (img_i_max - img_i_min) * 255

    return np.uint8(img)


# 单尺度Retinex，sigma = 300
def singleScaleRetinex(img, sigma):
    _temp = cv2.GaussianBlur(img, (0, 0), sigma)
    gaussian = np.where(_temp == 0, 0.01, _temp)  # 避免出现0值
    img_ssr = np.log10(img + 0.01) - np.log10(gaussian)  # 按照公式计算

    return restore(img_ssr)


# 多尺度Retinex，sigma_list = [15, 80, 250]
def multiScaleRetinex(img, sigma_list):
    retinex = np.zeros_like(img * 1.0)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)
    retinex /= len(sigma_list)

    return restore(retinex)


def colorRestoration(img, alpha, beta):
    img = np.float32(img)
    img_sum = np.sum(img, axis=2, keepdims=True)
    return beta * (np.log10(img * alpha) - np.log10(img_sum))


def msrcr(img, sigma_list, alpha, beta, G, b):
    img_msr = multiScaleRetinex(img, sigma_list)
    img_color = colorRestoration(img, alpha, beta)
    img_msrcr = G * img_color * img_msr + b

    return restore(img_msrcr)


if __name__ == '__main__':
    sigma_list = [15, 80, 250]
    img = cv2.imread('data/tree.jpg')
    cv2.imshow('src', img)
    cv2.imshow('ssr', singleScaleRetinex(img, 300))
    cv2.imshow('msr', multiScaleRetinex(img, sigma_list))
    cv2.imshow('msrcr', msrcr(img, sigma_list, 125, 46, 5, 25))

    cv2.waitKey(0)
