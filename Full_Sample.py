import bpy
import numpy as np
from bpy import data as D
from bpy import context as C
from mathutils import *
from math import *
import blensor
import os

# 旋转矩阵


def rotate(x_ang, y_ang, z_ang):
    x_mat = np.mat([[1, 0, 0, 0],
                    [0, cos(x_ang), sin(x_ang), 0],
                    [0, -sin(x_ang), cos(x_ang), 0],
                    [0, 0, 0, 1]])
    y_mat = np.mat([[cos(y_ang), 0, -sin(y_ang), 0],
                    [0, 1, 0, 0],
                    [sin(y_ang), 0, cos(y_ang), 0],
                    [0, 0, 0, 1]])
    z_mat = np.mat([[cos(z_ang), sin(z_ang), 0, 0],
                    [-sin(z_ang), cos(z_ang), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    rot_mat = x_mat*y_mat*z_mat
    return rot_mat


path = "C:/Users/bibiz/Documents/LiDAR-Scan-Simulation/data.csv"
"""设定Velodyne HDL 64E2激光雷达参数"""
scanner = bpy.data.objects["Camera"]
scanner.location = (0, -6, 0)
# scanner.scan_type = "tof";
scanner.scan_type = "velodyne"
scanner.velodyne_model = "hdl64e2"
scanner.velodyne_angle_resolution = 0.17
scanner.velodyne_rotation_speed = 10
scanner.velodyne_max_dist = 120
scanner.velodyne_ref_dist = 50
scanner.velodyne_ref_limit = 0.1
scanner.velodyne_ref_slope = 0.01
scanner.velodyne_start_angle = -30
scanner.velodyne_end_angle = 30

"""删除之前所有的扫描结果"""
for item in bpy.data.objects:
    if item.type == 'MESH' and item.name.startswith('Scan'):
        bpy.data.objects.remove(item)

"""设定船体的位置和欧拉角"""
Destroyer = bpy.data.objects["DDG"]
Destroyer.rotation_euler = (0, 0, 0)
Destroyer.location = (0, 0, 0)

"""删除上次扫描保存的点云文件，并创建新文件"""
try:
    os.remove(path)
except:
    pass

"""删除Blensor视窗中的所有扫描结果并启动新扫描"""
bpy.ops.blensor.delete_scans()
bpy.ops.blensor.scan()
f = open(path, "a+")

"""平移船体并进行全船扫描"""
for z_loc in range(-1, 0):
    for x_loc in range(-8, 8):
        # clear all scanning datas
        for item in bpy.data.objects:
            if item.type == 'MESH' and item.name.startswith('Scan'):
                bpy.data.objects.remove(item)

        Destroyer.location = (x_loc, 0, z_loc)
        bpy.ops.blensor.delete_scans()
        bpy.ops.blensor.scan()

        for item in bpy.data.objects:
            if item.type == 'MESH' and item.name.startswith('Scan'):
                x_ang = item.rotation_euler[0]
                y_ang = item.rotation_euler[1]
                z_ang = item.rotation_euler[2]
                rot_mat = rotate(x_ang, y_ang, z_ang)  # 旋转矩阵

                for sp in item.data.vertices:
                    xyz = np.mat([sp.co[0], sp.co[1], sp.co[2], 1.0])
                    xyz_rot = xyz*rot_mat

                    str = '%#5.3f,%#5.3f,%#5.3f \n' % (
                        xyz_rot[0, 0]+item.location[0]-x_loc, xyz_rot[0, 1]+item.location[1], xyz_rot[0, 2]+item.location[2]-z_loc)
                    f.write(str)

f.close()
