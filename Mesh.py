import open3d as o3d
import numpy as np
from tabulate import tabulate

path_obj = 'DDG055.stl'  # 模型路径，支持后缀：stl/ply/obj/off/gltf/glb
"""读入网格模型"""
mesh = o3d.io.read_triangle_mesh(path_obj)
"""计算网格顶点"""
mesh.compute_vertex_normals()

"""创建轴对齐包围盒并可视化网格模型"""
box1 = mesh.get_axis_aligned_bounding_box()
box1.color = (1, 0, 0)  # 红色
o3d.visualization.draw_geometries([mesh, box1])
bound_point1 = np.asarray(box1.get_box_points()).transpose()

hull_stem_bound = max(bound_point1[0])  # 船艏坐标
hull_stern_bound = min(bound_point1[0])  # 船尾坐标
hull_left_bound = min(bound_point1[1])  # 船左舷坐标
hull_right_bound = max(bound_point1[1])  # 船右舷坐标
hull_top_bound = max(bound_point1[2])  # 船顶坐标
hull_bottom_bound = min(bound_point1[2])  # 船底坐标

table = [['船首边界坐标', hull_stem_bound],
         ['船尾边界坐标', hull_stern_bound],
         ['船左舷边界坐标', hull_left_bound],
         ['船右舷边界坐标', hull_right_bound],
         ['船底部边界坐标', hull_bottom_bound],
         ['船顶部边界坐标', hull_top_bound]]

print(tabulate(table, headers=["名称", "值"]))
