import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(3,16,5) #修改卷积核大小为5*5
        self.pool1 = nn.MaxPool2d(2,2) #修改池化层步长为2
        self.conv2 = nn.Conv2d(16,32,5) #修改卷积核大小为5*5
        self.pool2 = nn.MaxPool2d(2,2) #修改池化层步长为2
        self.fc1 = nn.Linear(32*61*61,120) #修改全连接层输入维度为32*61*61
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.selu(self.conv1(x))#input(3,256,256) output(16,252,252)
        x = self.pool1(x) #output(16,126,126)
        x = F.relu(self.conv2(x)) #output(32,122,122)
        x = self.pool2(x) #output(32,61,61)
        x = x.view(-1,32*61*61) #output(32*61*61)
        x = F.selu(self.fc1(x)) #output(120)
        x = F.selu(self.fc2(x)) #output(84)
        x = self.fc3(x) #output(10)
        return x

# #model调试
# import torch
# #定义shape
# input1 = torch.rand([32,3,256,256])
# model = LeNet()#实例化
# print(model)
# #输入网络中
# output = model(input1)
