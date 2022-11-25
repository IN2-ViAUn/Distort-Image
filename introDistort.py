import os.path
import numpy as np
import cv2 as cv
from virtualCam import VirtualCamera, MeshGen
from fisheyeCam import FishEyeGenerator
import time

images_root = "D://New_Pycharm_Project//Distorted//images"

"""
    两种径向畸变是在virtualCam中做的
    鱼眼镜头的畸变是在fisheyeCam中做的
    
    因为畸变的代码实现方式不同，所以在两个文件下
    其中virtualCam其实是可以实现任何的一种畸变的，也包括鱼眼畸变，
    但是fisheyeCam是基于"Universal Semantic Segmentation for Fisheye Urban Driving Images" 实现的鱼眼畸变， 可能更容易让人接受
"""


def fisheye_distortion(src_list, focal=200, alpha=0, beta=0, gamma=0, tx=0, ty=0, tz=-1):
    """
    实现鱼眼变换
    :param src_list: 图片
    :param focal: 焦距
    :param alpha: x轴旋转角 [0·360]
    :param beta: y周旋转角 [0·360]
    :param gamma: z周旋转角 [0·360]
    :param tx: x轴位移，建议在[-0.6, 0.6]
    :param ty: y轴位移，建议在[-0.6, 0.6]
    :param tz: z轴位移，建议在[-0.6, 0.6]
    :return: numpy-变换后的图像
    """
    dst_list = []

    for src in src_list:
        # 创建虚拟相机，默认畸变后的图和原始图像一样大
        fish_distort = FishEyeGenerator(focal, [src.shape[0], src.shape[1]])
        # 设置外置参数，z怎么设置意义不大
        fish_distort.set_ext_params([alpha, beta, gamma, tx, ty, -1])
        fish_distort.print_ext_param()
        # 引入畸变
        dst = fish_distort.transFromColor(src, reuse=False)
        dst_list.append(dst)

    return dst_list


def barral_distortion(src_list, focal=200, alpha=0, beta=0, gamma=0, tx=0, ty=0, tz=0, b1=1, b2=0.5, b3=0.1):
    """
    实现桶形畸变
    :param src_list: 图片
    :param focal: 焦距
    :param alpha: x轴旋转角 [0·360]
    :param beta: y周旋转角 [0·360]
    :param gamma: z周旋转角 [0·360]
    :param tx: x轴位移，建议在[-0.6, 0.6]
    :param ty: y轴位移，建议在[-0.6, 0.6]
    :param tz: z轴位移，建议在[-0.6, 0.6]
    :param b1: 一阶畸变参数：[0~1]
    :param b2: 二阶畸变参数: [0~1]
    :param b3:  三阶畸变参数: 0
    :return:
    """
    dst_list = []

    for src in src_list:
        distort = VirtualCamera(dst_h=src.shape[0], dst_w=src.shape[1], src_shape=src.shape)
        # 设置参数
        distort.set_rvec(alpha, beta, gamma)
        distort.set_tvec(tx, ty, tz)
        distort.set_barral_dist(b1, b2, b3)
        # 生成图片
        dst2 = distort.transFromColor(src, focal, reuse=False)
        dst_list.append(dst2)

    return dst_list


def pincushion_distortion(src_list, focal=200, alpha=0, beta=0, gamma=0, tx=0, ty=0, tz=0, p1=-0.07, p2=0.0):
    """
    实现枕形畸变
    :param src_list: 图片
    :param focal: 焦距
    :param alpha: x轴旋转角 [0·360]
    :param beta: y周旋转角 [0·360]
    :param gamma: z周旋转角 [0·360]
    :param tx: x轴位移，建议在[-0.6, 0.6]
    :param ty: y轴位移，建议在[-0.6, 0.6]
    :param tz: z轴位移，建议在[-0.6, 0.6]
    :param p1: 一阶畸变参数：[-0.1~0]
    :param p2: 二阶畸变参数: 0
    :return:
    """
    dst_list = []

    for src in src_list:
        distort = VirtualCamera(dst_h=src.shape[0], dst_w=src.shape[1], src_shape=src.shape)
        # 设置参数
        distort.set_rvec(alpha, beta, gamma)
        distort.set_tvec(tx, ty, tz)
        distort.set_Pillow_dist(p1, p2)
        # 生成图片
        dst3 = distort.transFromColor(src, 200, reuse=False)
        dst_list.append(dst3)

    return dst_list


def funny_mirror(path):
    pass


def test_distortion(img_path):
    # 读取图片
    src = cv.imread(os.path.join(images_root, img_path))
    cv.imshow("src", src)
    H, W = src.shape[:2]

    # 鱼眼变换
    fisheye_distort = FishEyeGenerator(200, [H, W])
    fisheye_distort.set_ext_params([0, 0, 0, 0, 0, -1])
    dst1 = fisheye_distort.transFromColor(src, reuse=False)
    cv.imshow("fisheye_dst", dst1)

    # 桶形畸变
    barral_distort = VirtualCamera(dst_h=H, dst_w=W, src_shape=src.shape)
    barral_distort.set_barral_dist(1, 0.5, 0.1)
    dst2 = barral_distort.transFromColor(src, 200, reuse=False)
    cv.imshow("barral_dst", dst2)

    # 枕形畸变
    pincushion_distort = VirtualCamera(dst_h=H, dst_w=W, src_shape=src.shape)
    pincushion_distort.set_Pillow_dist(-0.07, 0.0)
    dst3 = pincushion_distort.transFromColor(src, 200, reuse=False)
    cv.imshow("pincushion_dst", dst3)

    cv.waitKey(0)


# 测试images下的某一张图片
test_distortion("00000.png")
