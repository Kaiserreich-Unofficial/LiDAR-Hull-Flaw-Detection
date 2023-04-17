import numpy as np
import open3d as o3d

np.set_printoptions(suppress=True)  # 取消默认的科学计数法
Data1 = np.loadtxt('data.csv', dtype=np.float64,
                   delimiter=',', usecols=(0, 1, 2), unpack=False)
Data1 = Data1 - np.mean(Data1, axis=0)  # 坐标归一化

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(Data1)
print(np.asarray(pcd.points))

o3d.visualization.draw_geometries([pcd], window_name="初始点云")
