import numpy as np
import open3d as o3d
import os
from functools import partial

Model = 'DDG055.stl'


def Hull_Points_Cut(pcd):
    """点云切割函数"""
    cloud_points_array = np.asarray(pcd.points)
    mask = (cloud_points_array[:, 0] > -87) & (cloud_points_array[:, 0] < 90) & \
           (cloud_points_array[:, 1] < 0) & (cloud_points_array[:, 2] > 0) & \
           (cloud_points_array[:, 2] < 7)
    cloud_points_array = cloud_points_array[mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_points_array)
    return pcd


def Hull_Vertices_Cut(mesh):
    """重建曲面边界切割函数"""
    # 创建一个轴对齐的边界框对象
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-87, -12, 0), max_bound=(90, 0, 7))
    cropped_mesh = mesh.crop(bbox)
    return cropped_mesh


def Bound_Box(obj, obj_name, color, display_model=True):
    """创建轴对齐包围盒并可视化网格模型"""
    box = obj.get_axis_aligned_bounding_box()  # 创建包围盒
    box.color = color  # color = (1, 0, 0)
    if display_model:
        o3d.visualization.draw_geometries(
            [obj, box], window_name=obj_name+"边界包围盒")

    bound_point = np.asarray(box.get_box_points()).transpose()

    obj_stem_bound = max(bound_point[0])  # 目标艏坐标
    obj_stern_bound = min(bound_point[0])  # 目标尾坐标
    obj_left_bound = min(bound_point[1])  # 目标左舷坐标
    obj_right_bound = max(bound_point[1])  # 目标右舷坐标
    obj_top_bound = max(bound_point[2])  # 目标顶坐标
    obj_bottom_bound = min(bound_point[2])  # 目标底坐标
    return [obj_stem_bound, obj_stern_bound, obj_left_bound, obj_right_bound, obj_top_bound, obj_bottom_bound]


def Hidden_Spots_Removal(pcd, param):
    """隐点去除(Katz等, 2007)"""
    # 定义隐点去除的参数(视点位置、反演圆半径)
    camera = [0, param[2] * 5, 0]  # 定义用于隐藏点删除的参数，获取从给定视图中可见的所有点，可视化结果
    radius = abs(param[2])*20
    # 点云反演
    _, pt_map = pcd.hidden_point_removal(camera, radius)
    # 选择移除后的内点
    pcd = pcd.select_by_index(pt_map)
    return pcd


def Fitness_Estm(source, target,
                 threshold=0.1,  # 最大的对应点-对的距离
                 ):
    """估计扫描后的点云与原模型表面的拟合程度(以RMSE计)"""
    trans_init = np.identity(4)
    reg_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init)
    return reg_icp.fitness, reg_icp.inlier_rmse


def Estm_Normals(pcd, radius=1.0, max_nn=100):
    """估算法线"""
    radius = 1.0  # 搜索半径
    max_nn = 100   # 邻域内用于估算法线的最大点数
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    # 法线与y轴负方向对齐(Disabled)
    # o3d.geometry.PointCloud.orient_normals_to_align_with_direction(
    #    pcd, orientation_reference=np.array([0.0, -1.0, 0.0]))
    return pcd


def Possion_Reshape(pcd):
    """泊松表面重建"""
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)
    return mesh


def Possion_Sample(mesh, n=100000):
    """体素均匀表面采样"""
    return mesh.sample_points_uniformly(number_of_points=n)


print("读取原船体模型并计算网格")
if os.path.exists(Model):
    """读入网格模型"""
    mesh = o3d.io.read_triangle_mesh(Model)
    """计算网格顶点"""
    mesh.compute_vertex_normals()
else:
    print("船体原模型不存在，请将%s放到该文件夹下!".format(Model))
    exit(-1)
if os.path.exists("Scan_Result.pcd"):
    pcd = o3d.io.read_point_cloud("Scan_Result.pcd")
else:
    if os.path.exists("data.csv"):
        os.system("python CSV2PCD.py")
        pcd = o3d.io.read_point_cloud("Scan_Result.pcd")
    else:
        print("扫描数据不存在，请先在Blensor中启动扫描!")
        exit(-1)

# print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd], window_name="初始点云")

mesh.scale(1, center=(0, 0, 0))
pcd_param = Bound_Box(pcd, "点云", (1, 0, 0), False)
mesh_param = Bound_Box(mesh, "船体", (0, 1, 0))

scale = mesh_param[0]/pcd_param[0]

print("放大点云至原模型大小")
pcd.scale(scale, center=pcd.get_center())
pcd_param = Bound_Box(pcd, "点云", (1, 0, 0))
print("平移点云至包围盒边界与原模型对齐")
pcd.translate((0, mesh_param[2] - pcd_param[2],
               mesh_param[5] - pcd_param[5]), relative=True)

o3d.visualization.draw_geometries([pcd, mesh], window_name="扫描生成点云与原模型比较")

print("对mesh后船体模型进行体素均匀采样(100000点)")
pcd2 = Possion_Sample(mesh)
print("对原点云和均匀采样的点云进行切割")
pcd = Hull_Points_Cut(pcd)
pcd2 = Hull_Points_Cut(pcd2)
print("处理均匀采样点云切割后残留的内部结构")
pcd2 = Hidden_Spots_Removal(pcd2, mesh_param)

pcd.paint_uniform_color([1, 0.706, 0])  # 原点云使用黄色绘制
pcd2.paint_uniform_color([0, 0.651, 0.929])  # 目标点云使用青色绘制

o3d.visualization.draw_geometries(
    [pcd, pcd2], window_name="Mesh后采样表面点云与LiDAR点云对比")

fitness, rmse = Fitness_Estm(pcd2, pcd)
print("拟合准度指标:{0:.2%},异常点的均方根误差:{1:.3f}".format(fitness, rmse))

octree = o3d.geometry.Octree(max_depth=6)
octree.convert_from_point_cloud(pcd, size_expand=0.01)  # 最小体素大小 0.01cm

# o3d.visualization.draw_geometries([pcd, hull_ls], window_name="添加最小凸包")
o3d.visualization.draw_geometries([pcd, octree], window_name="八叉树生成")

# 定义一个回调函数，用于处理每个子叶结点


def f_downsample(n, node, node_info):
    global points
    # 如果子叶结点包含的点数超过 n，则随机选择 n 个点作为代表
    if len(node.indices) > n:
        pindex = np.random.choice(node.indices, size=n, replace=False)
        for index in pindex:
            points.append(pcd.points[index])
    # 返回False表示不需要提前停止遍历
    return False


default_downsample = partial(f_downsample, 100)

points = []
# 遍历树
octree.traverse(default_downsample)
points = np.asarray(points)
print("降采样后的点数量:{:,}".format(len(points)), "降采样前的点数量:{:,}".format(len(pcd.points)),
      "降采样压缩率:{:.2%}".format(len(points)/len(pcd.points)))

downsampled_pcd = o3d.geometry.PointCloud()
downsampled_pcd.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([downsampled_pcd], window_name="降采样后的点云")


downsampled_pcd = Estm_Normals(downsampled_pcd)
# 可视化
o3d.visualization.draw_geometries(
    [downsampled_pcd], point_show_normal=True, window_name="点云法线估计")

rec_mesh = Possion_Reshape(downsampled_pcd)
croped_mesh = Hull_Vertices_Cut(rec_mesh)

o3d.visualization.draw_geometries([croped_mesh], window_name="泊松重建并切割后的船体曲面")

print("对泊松重建表面进行点云重采样")
reshaped_pcd = Possion_Sample(croped_mesh)

fitness, rmse = Fitness_Estm(pcd2, reshaped_pcd)
print("拟合准度指标:{0:.2%},异常点的均方根误差:{1:.3f}".format(fitness, rmse))
