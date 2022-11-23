import os.path
import numpy as np
import cv2 as cv
from vcam import vcam, meshGen

def FunnyGen(path):
    img = cv.imread(path)
    # 获得图像宽高
    H, W = img.shape[:2]

    # 创建虚拟相机
    camera = vcam(H=H, W=W)

    # 设置参数
    camera.focus = 100
    camera.update_M()

    # 创建surface：将被处理的图像视为一个平的平面(no depth consideration for pixel)
    plane = meshGen(H, W)

    # wrapping the surface：可以构造鱼眼镜头
    # plane.Z -= 100*np.sqrt((plane.X*2.0/plane.W)**2+(plane.Y*2.0/plane.H)**2)
    plane.drawMesh("dst")

    # 获得三维坐标
    pts3d = plane.getPlane()

    # 设置桶形畸变
    camera.set_barral_dist(1, 0.1, 0.1)

    # 设置枕形畸变

    # 将三维坐标的点映射到像平面：依据Virtual Camera的参数
    pts2d = camera.project(pts3d)

    # 获得坐标映射
    map_x, map_y = camera.getMaps(pts2d)

    # inverse mapping:将图像映射
    # 这里也可以设置对应的border颜色，默认是黑色
    output = cv.remap(img, map_x, map_y, interpolation = cv.INTER_LINEAR)

    # flip：左右翻转
    output = cv.flip(output, 1)

    # display the img and output
    cv.imshow("src", img)
    cv.imshow('dst', output)
    cv.waitKey(0)

images_root = "D://New_Pycharm_Project//Distorted//images"
path = os.path.join(images_root, '00000.png')

FunnyGen(path)