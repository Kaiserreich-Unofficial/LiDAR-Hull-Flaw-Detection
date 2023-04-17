# **基于水下LiDAR的船体结构损伤检测**

鄙人的毕业设计程序副本

**注：本项目主要使用Open3D实现，遵循MIT开源协议**

## 第一部分：LiDAR扫描与3D重建

### 程序主体

1. CSV2PCD.py —— LiDAR扫描数据（CSV格式）的转换程序（转换为PCD格式）
2. Full_Sample.py —— Blensor的全船扫描程序
3. Part_Sample.py —— Blensor的部分船体扫描程序
4. Mesh.py —— SolidWorks模型和参数预览
5. Raw_Visualize.py —— 初始点云预览程序
6. Visualize.py —— 点云数据处理和可视化主程序
7. Compress-Fitness_Curve.py —— 八叉树压缩率-拟合准度曲线可视化程序

### 数据文件

1. data.csv —— Blensor扫描保存的数据文件
2. DDG055.stl —— 原船SolidWorks模型
3. DDG055.blend* —— 原船及Lidar Blender模型

### 程序环境

Blensor 1.0.18 RC 10 Windows

python == 3.10.10

numpy == 1.23.5

matplotlib == 3.7.1

open3d == 0.16.0

tabulate == 0.9.0

tqdm == 4.65.0

### 说明

Open3d未开启CUDA加速，若想使用CUDA数组加速请自行编译Open3D

关于matplotlib输出中文乱码问题，请自行百度并修改matplotlibrc

## 第二部分：船体损伤识别

### 程序主体

1. Image2PCD.py —— 标注后数据集转点云分类转换程序
2. Part_Crack_PCD_Reshape.py —— 裂纹板/钢板表面重建程序
3. PCD2GreyScale_Feature_Img.py —— 点云-灰度直方图转换程序

### 数据文件

1. Label_Image —— 标注后的KolektorSDD数据集
2. Fine_PCD —— 完好的钢板点云
3. Crack_PCD —— 带裂纹的钢板点云

### 程序环境

CUDA 11.7

torch == 2.0.0+cu117

torchvision == 0.15.1+cu117

### 说明

Fine_PCD 和 Crack_PCD 两个文件夹必须存在，否则程序无法正常运行（没有请自己创建）

裂纹板/钢板表面重建程序算法同第一部分中的曲面重建，主要用来验证重建算法的理论可行性
