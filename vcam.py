import cv2
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')


class vcam:
    def __init__(self, H=400, W=400):
        self.H = H  # Desired height of the frame of output video
        self.W = W  # Desired width of the frame of output

        self.ox = W // 2  # 相面中心坐标
        self.oy = H // 2

        self.alpha = math.radians(0)  # 相对世界坐标系的三个轴的旋转角度
        self.beta = math.radians(0)
        self.gamma = math.radians(0)

        self.Tx = 0  # 相对世界坐标系的平移
        self.Ty = 0
        self.Tz = 0

        self.K = 0  # camera coordinate 3D -> image 2D
        self.R = 0
        self.sh = 0  # Shere factor
        self.P = 0

        self.KpCoeff = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # k1,k2,k3,p1,p2, 径向畸变参数

        self.focus = 100  # Focal length of camera in mm, 假设两个方向相同，如果不同要修改K

        self.sx = 1  # Effective size of a pixel in mm
        self.sy = 1  # Effective size of a pixel in mm

        self.RT = None  # 世界坐标系->相机坐标系

        self.set_tvec(0, 0, -self.focus)
        self.update_M()

    def update_M(self):
        # 构建相机外参RT，世界坐标系->相机坐标系的变换
        Rx = np.array([[1, 0, 0], [0, math.cos(self.alpha), -math.sin(self.alpha)],
                       [0, math.sin(self.alpha), math.cos(self.alpha)]])
        Ry = np.array(
            [[math.cos(self.beta), 0, -math.sin(self.beta)], [0, 1, 0], [math.sin(self.beta), 0, math.cos(self.beta)]])
        Rz = np.array(
            [[math.cos(self.gamma), -math.sin(self.gamma), 0], [math.sin(self.gamma), math.cos(self.gamma), 0],
             [0, 0, 1]])
        self.R = np.matmul(Rx, np.matmul(Ry, Rz))
        self.K = np.array([[-self.focus / self.sx, self.sh, self.ox], [0, self.focus / self.sy, self.oy], [0, 0, 1]])
        self.M1 = np.array([[1, 0, 0, -self.Tx], [0, 1, 0, -self.Ty], [0, 0, 1, -self.Tz]])
        self.RT = np.matmul(self.R, self.M1)

    def project(self, src):
        """
        实现从世界坐标系到像素坐标系的变换
        :param src: 世界坐标系[4, coordinate numbers]
        :return: 像素坐标系：
        """

        # 世界坐标系->相机坐标系。
        # RT：[3,4]
        # src: [4, coordinate numbers]
        # pts2d: [3, coordinate numbers]
        pts2d = np.matmul(self.RT, src)

        # 相机坐标系->像素坐标系(同时引入径向畸变)
        try:
            # 转为二维平面
            x_1 = pts2d[0, :] * 1.0 / (pts2d[2, :] + 0.0000000001)
            y_1 = pts2d[1, :] * 1.0 / (pts2d[2, :] + 0.0000000001)

            # 计算畸变泰勒公式的半径值
            x_2 = x_1 ** 2
            y_2 = y_1 ** 2
            x_y = x_1 * y_1
            r_2 = x_2 + y_2
            r_4 = r_2 ** 2
            r_6 = r_2 ** 3

            # 计算桶形畸变参数K
            K = (1 + self.KpCoeff[0] * r_2 + self.KpCoeff[1] * r_4 + self.KpCoeff[2] * r_6)

            # 将桶形畸变和枕型畸变插入，求出畸变的坐标(dis_x, dis_y)
            dis_x = x_1 * K + 2 * self.KpCoeff[3] * x_y + self.KpCoeff[4] * (r_2 + 2 * x_2)
            dis_y = y_1 * K + 2 * self.KpCoeff[4] * x_y + self.KpCoeff[3] * (r_2 + 2 * y_2)

            # camera coordinate -> image coordinate
            x = self.K[0, 0] * dis_x + self.K[0, 2]
            y = self.K[1, 1] * dis_y + self.K[1, 2]
        except:
            # 除法只会发生在转为二维平面的地方，在构建你的三维畸变平面时不要出现z=0
            print("Division by zero!")
            x = pts2d[0, :] * 0
            y = pts2d[1, :] * 0

        return np.concatenate(([x], [y]))

    def set_tvec(self, x, y, z):
        """
        设置坐标系平移参数
        """
        self.Tx = x
        self.Ty = y
        self.Tz = z
        self.update_M()

    def set_rvec(self, alpha, beta, gamma):
        """
        设置坐标系旋转参数
        """
        self.alpha = (alpha / 180.0) * np.pi
        self.beta = (beta / 180.0) * np.pi
        self.gamma = (gamma / 180.0) * np.pi
        self.update_M()

    def set_barral_dist(self, k1, k2, k3):
        """
        设置桶形畸变参数
        """
        self.KpCoeff[0] = k1
        self.KpCoeff[1] = k2
        self.KpCoeff[2] = k3

    def set_Pillow_dist(self, p1, p2):
        """
        设置枕形畸变参数
        """
        self.KpCoeff[3] = p1
        self.KpCoeff[4] = p2

    def getMaps(self, pts2d):
        """
        获得真实图像到畸变图像的映射矩阵
        """
        pts1, pts2 = np.split(pts2d, 2)
        x = pts1.reshape(self.H, self.W)
        y = pts2.reshape(self.H, self.W)
        return x.astype(np.float32), y.astype(np.float32)

    # 下面两个函数的功能在meshGen中实现
    def renderMesh(self, src):
        self.update_M()
        pts = self.project(src)
        canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        pts = (pts.T).reshape(-1, 1, 2).astype(np.int32)
        cv2.drawContours(canvas, pts, -1, (0, 255, 0), 3)
        return canvas

    def applyMesh(self, img, meshPts):
        pts1, pts2 = np.split(self.project(meshPts), 2)
        x = pts1.reshape(self.H, self.W)
        y = pts2.reshape(self.H, self.W)
        return cv2.remap(img, x.astype(np.float32), y.astype(np.float32), interpolation=cv2.INTER_LINEAR)


class meshGen:
    def __init__(self, H, W):
        self.H = H
        self.W = W

        # 将物面进行切分，分别在x和y方向上
        x = np.linspace(-self.W / 2, self.W / 2, self.W)
        y = np.linspace(-self.H / 2, self.H / 2, self.H)

        # 依据切分来构造网格，xv和yv分别代表了从axis=0时和axis=1方向的坐标
        xv, yv = np.meshgrid(x, y)
        self.mesh_shape = xv.shape

        # 将meshgrid转换成坐标。之间的对应关系(X, Y)。初始化的物面是在Z=1的面
        self.X = xv.reshape(-1, 1)
        self.Y = yv.reshape(-1, 1)
        self.Z = self.X * 0 + 1  # The mesh will be located on Z = 1 plane

        # 绘制meshgrid
        self.drawMesh()

    def getPlane(self):
        return np.concatenate(([self.X], [self.Y], [self.Z], [self.X * 0 + 1]))[:, :, 0]

    def drawMesh(self, title="src"):
        # 依据传入的lambda函数来求出z
        x = self.X.reshape(self.mesh_shape)
        y = self.Y.reshape(self.mesh_shape)
        z = self.Z.reshape(self.mesh_shape)
        # 绘制3D
        plt.style.use('_mpl-gallery')
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax.plot_wireframe(x, y, z, rstride=1, cstride=1, cmap=matplotlib.cm.Blues)
        ax.plot_surface(x, y, z, cmap=matplotlib.cm.Blues)
        ax.set(xticklabels=[],
               yticklabels=[],
               zticklabels=[])
        fig.suptitle(title)
        # ax.set_title(title, loc="center")
        plt.show()
