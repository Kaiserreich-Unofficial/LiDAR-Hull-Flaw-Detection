import torch.nn as nn
import torch.nn.functional as F
import torch

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 输入通道为1(灰度图)，输出通道为6，卷积核大小为5
        self.pool1 = nn.MaxPool2d(2, 2) # 池化核大小为2，步长为2
        self.conv2 = nn.Conv2d(6, 16, 5) # 输入通道为6，输出通道为16，卷积核大小为5
        self.pool2 = nn.MaxPool2d(2, 2) # 池化核大小为2，步长为2
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 全连接层，输入维度为16*5*5，输出维度为120
        self.fc2 = nn.Linear(120, 84) # 全连接层，输入维度为120，输出维度为84
        self.fc3 = nn.Linear(84, 10) # 全连接层，输入维度为84，输出维度为10

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 输入大小为32x32，输出大小为28x28
        x = self.pool1(x) # 输出大小为14x14
        x = F.relu(self.conv2(x)) # 输出大小为10x10
        x = self.pool2(x) # 输出大小为5x5
        x = x.view(-1, 16 * 5 * 5) # 展平成一维向量
        x = F.relu(self.fc1(x)) # 输出维度为120
        x = F.relu(self.fc2(x)) # 输出维度为84
        x = self.fc3(x) # 输出维度为10
        return x

# #定义shape
# input1 = torch.rand([32,1,32,32])
# model = LeNet()#实例化
# print(model)
# #输入网络中
# output = model(input1)
