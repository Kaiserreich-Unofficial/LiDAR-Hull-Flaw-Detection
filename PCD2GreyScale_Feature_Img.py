import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

pcd = o3d.io.read_point_cloud("Crack_PCD/1.pcd")

def main(pcd):
    point_cloud = np.asarray(pcd.points)

    # 将点云的坐标和高度值分别存储在两个一维数组中
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # 使用numpy.histogram2d函数，将x, y数组作为输入，指定bins参数为你想要的图片尺寸，指定weights参数为z数组，得到一个二维数组matrix，表示每个像素的灰度值
    matrix, _, _ = np.histogram2d(x, y, bins=(1267,1267))

    matrix = exposure.rescale_intensity(matrix, in_range=(0, 1), out_range=(0, 255))
    # print(matrix)

    # 使用matplotlib.pyplot.imshow函数，将matrix作为输入，显示或保存灰度图片
    plt.imshow(matrix, cmap='gray')
    plt.show()

if __name__ == "__main__":
    main(pcd)
