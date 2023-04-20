# -*- coding:utf-8 -*-
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import os
from PIL import Image


def save_xml(image_name, width, height, channel=3):
    save_dir = 'Fine_Annotations'

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'Fine_PCD'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = 'Unknown'

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % 1

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)

    save_xml = os.path.join(save_dir, image_name.replace('jpg', 'xml'))
    with open(save_xml, 'wb') as f:
        f.write(xml)

    return


if __name__ == '__main__':

    root_path = 'Fine_PCD'

    imagelist = os.listdir(os.path.join(root_path))

    for image_name in imagelist:
        if '.pcd' in image_name:  # 跳过苹果电脑生成的.DS_Store文件
            continue
        image = root_path + "\\"+ image_name  # 图片的路径
        img = Image.open(image)
        img_size = img.size
        w = img.width
        h = img.height
        print('image', image, w, h)

        save_xml(image_name, w, h)
