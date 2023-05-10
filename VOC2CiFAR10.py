import pickle
import numpy as np
from PIL import Image
import os
from easygui import diropenbox
import xml.etree.ElementTree as ET  # 导入xml库

def voc_to_cifar(voc_dir):
    # voc_dir is the directory of the VOC2007 dataset
    # return two tuples: (train_x, train_y) and (test_x, test_y)
    # train_x and test_x are numpy arrays of shape (n, 32, 32, 3), where n is the number of samples
    # train_y and test_y are numpy arrays of shape (n,), where n is the number of samples
    # the values of train_y and test_y are integers representing the labels

    # read the file names from train.txt and test.txt
    train_list = []
    test_list = []
    with open(os.path.join(voc_dir, "ImageSets", "Main", "train.txt"), "r") as f:
        for line in f:
            train_list.append(line.strip())
    with open(os.path.join(voc_dir, "ImageSets", "Main", "test.txt"), "r") as f:
        for line in f:
            test_list.append(line.strip())

    # initialize the arrays for features and labels
    train_x = np.empty((len(train_list), 256, 256, 3), dtype=np.uint8)
    train_y = np.empty((len(train_list),), dtype=np.int64)
    test_x = np.empty((len(test_list), 256, 256, 3), dtype=np.uint8)
    test_y = np.empty((len(test_list),), dtype=np.int64)

    # create a dictionary for label mapping
    label_dict = {"crack": 0}

    # loop through the train list and convert each image and label to numpy array
    for i, file_name in enumerate(train_list):
        # open the image and resize it to 256x256 pixels
        img = Image.open(os.path.join(
            voc_dir, "JPEGImages", file_name + ".jpg"))
        img = img.resize((256, 256))
        # convert the image to numpy array and add it to the feature array
        img_array = np.array(img)
        train_x[i] = img_array
        # read the label from the annotation file and convert it to a number
        # parse the xml file and get the label name
        tree = ET.parse(os.path.join(
            voc_dir, "Annotations", file_name + ".xml"))
        root = tree.getroot()
        label_name = root.find("object").find("name").text
        # map the label name to a number using the dictionary
        label = label_dict[label_name]
        train_y[i] = label

    # loop through the test list and do the same thing
    for i, file_name in enumerate(test_list):
        img = Image.open(os.path.join(
            voc_dir, "JPEGImages", file_name + ".jpg"))
        img = img.resize((256, 256))
        img_array = np.array(img)
        test_x[i] = img_array
        tree = ET.parse(os.path.join(
            voc_dir, "Annotations", file_name + ".xml"))
        root = tree.getroot()
        label_name = root.find("object").find("name").text
        label = label_dict[label_name]
        test_y[i] = label

    # return the two tuples of features and labels
    return (train_x, train_y), (test_x, test_y)


def cifar_to_pickle(train_x, train_y, test_x, test_y):
    # train_x and test_x are numpy arrays of shape (n, 256, 256, 3), where n is the number of samples
    # train_y and test_y are numpy arrays of shape (n,), where n is the number of samples
    # the values of train_y and test_y are integers representing the labels
    # generate three pickle files: batches.meta, data_batch_0 and test_batch

    # create a dictionary for meta data
    meta = {}
    meta["num_cases_per_batch"] = len(train_x)
    # change the label names to match the labels
    meta["label_names"] = ["crack"]
    meta["num_vis"] = 256 * 256 * 3

    # save the meta data as batches.meta file
    with open("batches.meta", "wb") as f:
        pickle.dump(meta, f)

    # create a dictionary for train data
    train = {}
    # flatten the images to one-dimensional arrays
    train["data"] = train_x.reshape((len(train_x), -1))
    train["labels"] = train_y.tolist()  # convert the labels to a list

    # save the train data as data_batch_0 file
    with open("data_batch_0", "wb") as f:
        pickle.dump(train, f)

    # create a dictionary for test data
    test = {}
    # flatten the images to one-dimensional arrays
    test["data"] = test_x.reshape((len(test_x), -1))
    test["labels"] = test_y.tolist()  # convert the labels to a list

    # save the test data as test_batch file
    with open("test_batch", "wb") as f:
        pickle.dump(test, f)


if __name__ == "__main__":
    # call the first function and get the converted dataset
    voc_dir = diropenbox("请精确到VOC2007文件夹","打开VOC2007数据集")
    if not voc_dir:
        exit()
    (train_x, train_y), (test_x, test_y) = voc_to_cifar(voc_dir)
    # call the second function and generate the pickle files
    cifar_to_pickle(train_x, train_y, test_x, test_y)
