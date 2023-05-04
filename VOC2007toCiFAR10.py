import os
import cv2
import pickle
import numpy as np
import xml.etree.ElementTree as ET
# 导入 PIL 和 skimage 库
from PIL import Image

# 定义VOC2007格式的数据集的路径
voc_dir = "VOCdevkit/VOC2007"
# 定义CiFAR10格式的数据集的路径
cifar_dir = ""
# 定义CiFAR10格式的数据集的文件名
cifar_file = "data_batch"
# 定义CiFAR10格式的数据集的图像大小
cifar_size = 32
# 定义VOC2007格式的数据集的类别和对应的编号
voc_classes = {"crack": 0}
# 定义一个空列表，用于存储转换后的图像和标注
cifar_data = []

# 遍历VOC2007格式的数据集中的图像文件和标注文件
for image_file in os.listdir(os.path.join(voc_dir, "JPEGImages")):
    # 获取图像文件的完整路径
    image_path = os.path.join(voc_dir, "JPEGImages", image_file)
    # 获取图像文件对应的标注文件的完整路径
    annotation_path = os.path.join(voc_dir, "Annotations", image_file[:-4] + ".xml")
    # 读取图像文件为一个numpy数组
    image = cv2.imread(image_path)
    # 使用 PIL 库将 numpy 数组转换为 Image 对象
    image = Image.fromarray(image)
    # 使用 PIL 库缩放图像为32x32像素，并转换回 numpy 数组
    image = image.resize((cifar_size, cifar_size))
    image = np.array(image)
    # 读取标注文件为一个XML树对象
    annotation = ET.parse(annotation_path)
    # 获取标注文件中的根元素
    root = annotation.getroot()
    # 获取标注文件中的第一个对象元素
    object = root.find("object")
    # 获取对象元素中的类别名称
    class_name = object.find("name").text
    # 获取对象元素中的边界框坐标
    bndbox = object.find("bndbox")
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)
    # 计算边界框在缩放后图像中的坐标
    xmin = int(xmin * cifar_size / 1267)
    ymin = int(ymin * cifar_size / 1267)
    xmax = int(xmax * cifar_size / 1267)
    ymax = int(ymax * cifar_size / 1267)
    # 裁剪图像为边界框内部区域，并转换为 Image 对象
    if not xmax < 32 & ymax < 32:
        xmax = 31
        ymax = 31
    image = Image.fromarray(image[ymin:ymax, xmin:xmax])

    # 使用 PIL 库再次缩放裁剪后的图像为32x32像素，并转换回 numpy 数组
    image = image.resize((cifar_size, cifar_size))
    image = np.array(image)
    # 将类别名称转换为一个10维向量，其中只有对应的编号为1，其余为0
    label = np.zeros(10)
    label[voc_classes[class_name]] = 1
    # 将转换后的图像和标注添加到cifar_data列表中
    cifar_data.append((image, label))
# 将cifar_data列表保存为一个pickle文件，作为CiFAR10格式的数据集
with open(os.path.join(cifar_dir, cifar_file), "wb+") as f:
    pickle.dump(cifar_data, f)
