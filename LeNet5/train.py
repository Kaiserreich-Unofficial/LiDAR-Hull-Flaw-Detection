#train.py

import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np


#device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 50000张训练图片
train_set = torchvision.datasets.CIFAR10(root='./LeNet5', train=True,
                                         download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                           shuffle=False, num_workers=0)

# 10000张验证图片
val_set = torchvision.datasets.CIFAR10(root='./LeNet5', train=False,
                                       download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=4,
                                         shuffle=False, num_workers=0)
val_data_iter = iter(val_loader)
val_image, val_label = next(val_data_iter)
print(val_image.size())

net = LeNet()
net.to(device)

loss_function = nn.CrossEntropyLoss()
#定义优化器
optimizer = optim.Adam(net.parameters(), lr=1e-3)

#训练过程
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0 #累加损失
    for step, data in enumerate(train_loader, start=0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        #print(inputs.size(), labels.size())
        # zero the parameter gradients
        optimizer.zero_grad()#如果不清除历史梯度，就会对计算的历史梯度进行累加
        # forward + backward + optimize
        outputs = net(inputs.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if step % 500 == 499:    # print every 500 mini-batches
            with torch.no_grad():#上下文管理器
                outputs = net(val_image.to(device))  # [batch, 10]
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = (predict_y == val_label.to(device)).sum().item() / val_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print('Finished Training')

save_path = './LeNet5/Lenet.pth'
torch.save(net.state_dict(), save_path)
