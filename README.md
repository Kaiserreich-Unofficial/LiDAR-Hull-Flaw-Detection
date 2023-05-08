# **基于水下LiDAR的船体结构损伤检测**

鄙人的毕业设计程序副本

**注：本项目主要使用Open3D实现，遵循MIT开源协议**

## 第一部分：LiDAR扫描与3D重建

### 程序主体

1. CSV2PCD.py —— LiDAR扫描数据的转换（CSV2PCD）
2. Full_Sample.py —— Blensor的全船扫描
3. Mesh.py —— SolidWorks模型和参数预览
4. Raw_Visualize.py —— 初始点云预览
5. Visualize.py —— 点云数据处理和可视化
6. Compress-Fitness_Curve.py —— 八叉树压缩率-拟合准度曲线可视化

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

easygui == 0.98.3

### 说明

Open3d未开启CUDA加速，若想使用CUDA数组加速请自行编译Open3D

关于matplotlib输出中文乱码问题，请自行百度并修改matplotlibrc

## 第二部分：船体损伤识别

### 程序主体

1. Image2PCD.py —— 标注后数据集转点云分类转换
2. Part_Crack_PCD_Reshape.py —— 裂纹板/钢板表面重建
3. PCD2GreyScale_Feature_Img.py —— 点云-灰度直方图转换
4. voc_annotation.py —— VOC2007数据集索引生成（训练集\测试集）
5. LeNet5\model.py —— LeNet5模型class
6. LeNet5\train.py —— LeNet5分类器的训练程序
7. LeNet5\predict.py —— LeNet5分类器的测试程序

### 数据文件

1. Label_Image —— 标注后的KolektorSDD数据集（源用于图像分割）
2. Fine_PCD —— 完好的钢板点云及灰度直方图
3. Fine_Annotations —— 完好的钢板灰度直方图标签（空标签）
4. Crack_PCD —— 带裂纹的钢板点云及灰度直方图
5. Crack_Annotations —— 带裂纹的钢板灰度直方图标签
6. VOCdevkit —— 制作好的VOC2007数据集
7. batches.meta —— 转化的CiFAR10数据集meta索引文件
8. data_batch_0 —— CiFAR10训练集pickle文件
9. test_batch —— CiFAR10测试集pickle文件

### 程序环境

CUDA 11.7

torch == 2.0.0+cu117

torchvision == 0.15.1+cu117

labelImg == 1.8.6（可选，生成标签xml使用）

### 说明

项目使用的模拟数据集(Kolektor Surface-Defect Dataset)来自Vicos.si，感谢Tabernik等作者的辛勤付出

Fine_PCD 和 Crack_PCD 两个文件夹必须存在，否则程序无法正常运行（没有请自己创建）

裂纹板/钢板表面重建程序算法同第一部分中的曲面重建，主要用来验证重建算法的理论可行性

分类器使用LeNet5神经网络，数据集采用自制的VOC2007数据集转换为CiFAR10数据集进行分类器的训练和验证

VOC2007数据集请自行构建，利用直方图转换程序生成的jpg和labelimg画的标签复制到对应的文件夹中，训练集和测试集的索引可以利用voc_annotation.py完成

转换出的CiFAR10数据集不能使用pytorch的DataLoader进行加载，因为DataLoader的md5校验机制会阻止这一行为，故训练程序请使用我提供的train.py，利用pickle进行解包
