import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from multiprocessing.pool import Pool
import random
from copy import deepcopy

batch_size = 8
Crack_PCD = "Crack_PCD"
Fine_PCD = "Fine_PCD"


def clean():
    os.system("del Crack_PCD\*.jpg")
    os.system("del Fine_PCD\*.jpg")


def pcd2grayhistogram(file):
    pcd = o3d.io.read_point_cloud(file)
    name = os.path.splitext(file)[0].split("/")[-1]
    point_cloud = np.asarray(pcd.points)
    print("转换"+name+".pcd中，点云数量"+str(len(pcd.points)))

    # 将点云的坐标和高度值分别存储在两个一维数组中
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # 使用numpy.histogram2d函数，将x, y数组作为输入，指定bins参数为你想要的图片尺寸，指定weights参数为z数组，得到一个二维数组matrix，表示每个像素的灰度值
    matrix, _, _ = np.histogram2d(x, y, bins=(1267, 1267), weights=z)
    matrix = np.where((matrix > -0.01) & (matrix < 1), 0, 255)
    # matrix = exposure.rescale_intensity(
    #    matrix, in_range=(0, 1), out_range=(0, 255))
    # print(matrix)

    # 使用matplotlib.pyplot.imshow函数，将matrix作为输入，显示或保存灰度图片
    # plt.imshow(matrix, cmap='gray')
    # plt.show()
    plt.imsave(name+'.jpg', matrix, dpi=300, cmap='gray')
    return 0


def main(Source_Dir):
    # 定义存储文件名的数组
    file_names = []
    # 遍历文件夹，将文件名加入数组
    for filename in os.listdir(Source_Dir):
        file_names.append(os.path.join(Source_Dir, filename))
    print('Label Image Count:', len(file_names))
    count = len(file_names)
    with tqdm(total=count) as pbar:
        pbar.set_description('转化为灰度直方图中:')
        for i in range(int(count/batch_size) + 1):
            if len(file_names) > batch_size:
                selected_files = random.sample(file_names, batch_size)
                Update_Progress = batch_size
            else:
                selected_files = deepcopy(file_names)
                Update_Progress = len(file_names)
            file_list = []
            for file in selected_files:
                file_names.remove(file)
                file_list.append(file)

            pool = Pool(Update_Progress)
            result = pool.map(pcd2grayhistogram, file_list)
            if result == [0] * Update_Progress:
                pool.close()
            pool.join()

            pbar.update(Update_Progress)


if __name__ == "__main__":
    clean()
    print("转换有裂缝板点云为灰度直方图...")
    main(Crack_PCD)
    print("转换无裂缝板点云为灰度直方图...")
    main(Fine_PCD)
