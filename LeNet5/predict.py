import torch
from PIL import Image
import numpy as np
from model import LeNet
import pickle

# load the meta data file
with open("batches.meta", "rb") as f:
    meta = pickle.load(f)
# create an instance of the network
net = LeNet()
# load the model parameters
net.load_state_dict(torch.load("lenet5_model.pth"))

# define a function to convert an image to network input


# define a function to convert an image to network input
def image_to_input(image_path):
    # image_path is the path of the image file
    # return a torch tensor of shape (1, 3, 32, 32), where 1 is the batch size
    # open the image and resize it to 32x32 pixels
    img = Image.open(image_path)
    img = img.resize((32, 32))
    # convert the image to torch tensor and adjust its shape and type
    img_tensor = torch.from_numpy(np.array(img)).float()
    # change the order of dimensions from (H, W, C) to (C, H, W)
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0)  # add a dimension for batch size
    # return the tensor
    return img_tensor

# define a function to convert network output to label name and confidence


def output_to_label_and_confidence(output):
    # output is a torch tensor of shape (1, c), where 1 is the batch size and c is the number of classes
    # return a tuple of (label_name, confidence), where label_name is a string and confidence is a float
    # get the index and value of the maximum value in the output tensor
    value, index = torch.max(output, dim=1)
    index = index.item()  # convert the index to a python integer
    value = value.item()  # convert the value to a python float
    # get the label name from the meta data dictionary
    label_name = meta["label_names"][index]
    # get the confidence from the value
    if value >= 100:
        confidence = 100  # convert the probability to percentage
    else:
        confidence = value
    # return the tuple of label name and confidence
    return (label_name, confidence)


# read an image and convert it to network input
image_path = input("Path to Img:")
input_img = image_to_input(image_path)

# forward propagation and get the network output and label name
output = net(input_img)
label_name, confidence = output_to_label_and_confidence(output)

# print the image path and label name and confidence
print(f"Image: {image_path}, Label: {label_name}, Confidence: {confidence:.2f}%")
