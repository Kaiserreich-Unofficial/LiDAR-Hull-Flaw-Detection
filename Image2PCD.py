import open3d as o3d
from tqdm import tqdm
from PIL import Image
import numpy as np
import os
from multiprocessing.pool import Pool
import random
from copy import deepcopy

Source_Dir = "Label_Image"
Crack_PCD = "Flaw_PCD"
Fine_PCD = "Fine_PCD"

count = 0
batch_size = 8


def clean():
    os.system("del Flaw_PCD\*.pcd")
    os.system("del Fine_PCD\*.pcd")

def img2pcd(file):
    name = os.path.basename(file)[:-4]
    # 读取图片并转换为二值化的布尔数组
    img = Image.open(file)
    img = img.convert("L")
    img = np.array(img)
    img1 = np.where(img < 128, True, False)  # 根据条件选择元素
    img2 = np.where(img > 128, True, False)
    # 创建空的点云对象
    pcd = o3d.geometry.PointCloud()
    if any(np.nonzero(img2)[0]):
        path = Crack_PCD
    else:
        path = Fine_PCD

    # 找出所有黑色像素的坐标，并将它们作为点云的点
    x1, y1 = np.nonzero(img1)  # 返回非零元素的索引
    z1 = np.zeros_like(x1)  # 创建一个全零的数组，作为z坐标
    # 将三个数组沿着第二个维度堆叠起来，形成一个N*3的数组，并为前两个维度（X-Y平面）添加高斯噪声
    points1 = np.stack([x1 + np.random.normal(0, 1, size=x1.shape), y1 + np.random.normal(
        0, 1, size=y1.shape), z1], axis=1)

    # 找出所有白色像素的坐标，并将它们作为点云的点，并添加高斯噪声和反转z值
    x2, y2 = np.nonzero(img2)
    z2 = np.random.normal(0, 1, size=x2.shape)  # 创建一个高斯噪声数组，作为z坐标
    z2[z2 > 0] = -z2[z2 > 0]  # 将z值为正的点反转到负方向上
    # 将三个数组沿着第二个维度堆叠起来，形成一个N*3的数组，并添加高斯噪声
    points2 = np.stack([x2 + np.random.normal(0, 1, size=x2.shape), y2 + np.random.normal(
        0, 1, size=y2.shape), z2], axis=1)

    points = np.concatenate([points1, points2], axis=0)  # 将两个点云数组拼接起来
    # 将点云移动到以原点为中心
    points = points - np.mean(points, axis=0, keepdims=True)
    points = points.astype(np.float32)  # 转换为浮点类型
    pcd.points = o3d.utility.Vector3dVector(
        points)  # 将numpy数组转换为Open3D向量，并赋值给点云对象
    downpcd = pcd.voxel_down_sample(voxel_size=2)
    print("样本"+ name +",压缩后点数量:{0:},压缩前点数量:{1:},压缩率:{2:.2%}".format(
        len(downpcd.points), len(pcd.points), len(downpcd.points)/len(pcd.points)))
    o3d.io.write_point_cloud(path+"\\"+ name +".pcd", downpcd)
    return 0


if __name__ == "__main__":
    clean()
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
            result = pool.map(img2pcd, file_list)
            if result == [0] * Update_Progress:
                pool.close()
            pool.join()

            pbar.update(Update_Progress)
