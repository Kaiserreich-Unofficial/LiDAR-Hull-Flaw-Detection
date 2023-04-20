import open3d as o3d
from tqdm import tqdm
from PIL import Image
import numpy as np
import os
from multiprocessing.pool import Pool

Source_Dir = "Label_Image"
Crack_PCD = "Crack_PCD"
Fine_PCD = "Fine_PCD"

count = 0
batch_size = 8


def main_func(i):
    # 读取图片并转换为二值化的布尔数组
    img = Image.open(os.path.join(Source_Dir, str(i)+".bmp"))
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
    print("样本"+str(i)+",压缩后点数量:{0:},压缩前点数量压缩率:{1:},压缩率:{2:.2%}".format(
        len(downpcd.points), len(pcd.points), len(downpcd.points)/len(pcd.points)))
    o3d.io.write_point_cloud(path+"/"+str(i)+".pcd", downpcd)
    return 0


if __name__ == "__main__":
    # 遍历文件夹
    for path in os.listdir(Source_Dir):
        # 检查当前路径是否为文件
        if os.path.isfile(os.path.join(Source_Dir, path)):
            count += 1
    print('Label Image Count:', count)
    with tqdm(total=count) as pbar:
        pbar.set_description('转化点云中:')
        for i in range(1, int(count/batch_size)):
            numlist = []
            for j in range(batch_size):
                numlist.append(j + i*batch_size + 1)

            pool = Pool(batch_size)
            result = pool.map(main_func, numlist)
            if result == [0]*batch_size:
                pool.close()
            pool.join()

            pbar.update(batch_size)
