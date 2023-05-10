import torch
import torch.optim as optim
import torch.utils.data as data
import pickle
from model import LeNet
import numpy as np
import torch.cuda  # 导入torch.cuda模块

# define some hyperparameters
lr = 1e-3  # learning rate
bs = 32  # batch size
epochs = 20  # number of training epochs

# load the pickle files and convert them to torch dataset and dataloader
with open("batches.meta", "rb") as f:
    meta = pickle.load(f)
with open("data_batch_0", "rb") as f:
    train = pickle.load(f)
with open("test_batch", "rb") as f:
    test = pickle.load(f)

# convert the train features to torch tensor
train_x = torch.from_numpy(train["data"]).float()
# reshape the train features to match the network input shape
train_x = train_x.view(-1, 3, 256, 256)
# convert the train labels to torch tensor
train_y = torch.from_numpy(np.array(train["labels"])).long()
# convert the test features to torch tensor
test_x = torch.from_numpy(test["data"]).float()
# reshape the test features to match the network input shape
test_x = test_x.view(-1, 3, 256, 256)
# convert the test labels to torch tensor
test_y = torch.from_numpy(np.array(test["labels"])).long()

# create a torch dataset for train data
train_dataset = data.TensorDataset(train_x, train_y)
# create a torch dataloader for train data
train_loader = data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
# create a torch dataset for test data
test_dataset = data.TensorDataset(test_x, test_y)
# create a torch dataloader for test data
test_loader = data.DataLoader(test_dataset, batch_size=bs, shuffle=False)

# create an instance of the network
net = LeNet()
# define the loss function and the optimizer
# use cross entropy loss for classification task
criterion = torch.nn.CrossEntropyLoss()
# use stochastic gradient descent as the optimizer
optimizer = optim.Adam(net.parameters(), lr=lr)

# define a function to calculate accuracy


def accuracy(outputs, labels):
    # outputs is a tensor of shape (n, c), where n is the number of samples and c is the number of classes
    # labels is a tensor of shape (n,), where n is the number of samples
    # return a float representing the accuracy
    # get the predicted class for each sample
    _, preds = torch.max(outputs, dim=1)
    # count the number of correct predictions
    correct = (preds == labels).sum().item()
    total = len(labels)  # count the total number of samples
    acc = correct / total  # calculate the accuracy
    return acc

# 检查是否有可用的CUDA设备
if torch.cuda.is_available():  # 如果有可用的CUDA设备
    device = torch.device("cuda")  # 创建一个代表CUDA设备的对象
    print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")  # 打印使用的CUDA设备名称
else:  # 如果没有可用的CUDA设备
    device = torch.device("cpu")  # 创建一个代表CPU设备的对象
    print("Using CPU device")

# 将网络模型转移到CUDA设备上
net.to(device)

# 定义一个变量来记录最好的测试准确率，初始值为0.0
best_acc = 0.0

# start the training loop
for epoch in range(epochs):
    # initialize the running loss and accuracy for train data
    train_loss = 0.0
    train_acc = 0.0
    # loop through the train data
    for i, (inputs, labels) in enumerate(train_loader):
        # 将输入数据和标签转移到CUDA设备上
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward propagation
        outputs = net(inputs)
        # calculate the loss
        loss = criterion(outputs, labels)
        # backward propagation and parameter update
        loss.backward()
        optimizer.step()
        # accumulate the running loss and accuracy
        train_loss += loss.item()
        train_acc += accuracy(outputs, labels)

    # initialize the running loss and accuracy for test data
    test_loss = 0.0
    test_acc = 0.0
    # loop through the test data
    for inputs, labels in test_loader:
        # 将输入数据和标签转移到CUDA设备上
        inputs = inputs.to(device)
        labels = labels.to(device)
        # forward propagation
        outputs = net(inputs)
        # calculate the loss
        loss = criterion(outputs, labels)
        # accumulate the running loss and accuracy
        test_loss += loss.item()
        test_acc += accuracy(outputs, labels)

    # print the statistics for each epoch
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_acc / len(train_loader):.4f}, Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_acc / len(test_loader):.4f}")

    # 判断当前的测试准确率是否超过了最好的测试准确率
    if test_acc > best_acc:
        # 更新最好的测试准确率，并保存当前的网络模型为best_epoch.pth
        best_acc = test_acc
        torch.save(net.state_dict(), "best_epoch.pth")
        print(f"Saved best epoch model with accuracy: {best_acc:.4f}")

    # 每10个epoch结束后，保存当前的网络模型为lenet5_model_epoch_{epoch}.pth，其中{epoch}是当前的epoch编号
    if (epoch + 1) % 10 == 0:
        torch.save(net.state_dict(), f"lenet5_model_epoch_{epoch + 1}.pth")
        print(f"Saved epoch model with accuracy: {test_acc:.4f}")
