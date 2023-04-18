import numpy as np
import open3d as o3d
import os
from tabulate import tabulate
from easygui import fileopenbox


def Possion_Reshape(pcd):
    """泊松表面重建"""
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)
    return mesh


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


def Crack_Cut(mesh):
    """重建曲面边界切割函数"""
    # 创建一个轴对齐的边界框对象
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-635, -255, -5), max_bound=(635, 255, -0.01))
    cropped_mesh = mesh.crop(bbox)
    return cropped_mesh


def Board_Cut(mesh):
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-635, -255, -0.01), max_bound=(635, 255, 1))
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


def main(Model):
    print("读取原船体模型并计算网格")
    if os.path.exists(Model):
        pcd = o3d.io.read_point_cloud(Model)
    else:
        print("裂纹板点云不存在，请启动Image2PCD程序获取样本点云！")
        exit(-1)

    o3d.visualization.draw_geometries([pcd], window_name="初始点云")
    pcd = Estm_Normals(pcd)
    pcd_param = Bound_Box(pcd, "点云", (1, 0, 0), False)
    table = [['板前缘边界坐标', pcd_param[0]],
             ['板后缘边界坐标', pcd_param[1]],
             ['板左缘边界坐标', pcd_param[2]],
             ['板右缘边界坐标', pcd_param[3]],
             ['板底部边界坐标', pcd_param[4]],
             ['板顶部边界坐标', pcd_param[5]]]

    print(tabulate(table, headers=["名称", "值"]))

    mesh = Possion_Reshape(pcd)
    Crack_Mesh = Crack_Cut(mesh).paint_uniform_color([1, 0.706, 0])  # 橙色
    Board_Mesh = Board_Cut(mesh).paint_uniform_color([0.706, 1, 0])  # 橙色

    o3d.visualization.draw_geometries(
        [Crack_Mesh, Board_Mesh], window_name="重建后曲面")


if __name__ == "__main__":
    Model = fileopenbox(msg="曲面重建验证程序", title="打开点云文件",
                        default="*.pcd", filetypes=["*.pcd"])
    if not Model:
        exit(-1)
    main(Model)
