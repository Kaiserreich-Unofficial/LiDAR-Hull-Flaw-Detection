import torch
import torch.optim as optim
import torch.utils.data as data
import pickle
from model import LeNet
import numpy as np

# define some hyperparameters
lr = 1e-3  # learning rate
bs = 32  # batch size
epochs = 10  # number of training epochs

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
train_x = train_x.view(-1, 3, 32, 32)
# convert the train labels to torch tensor
train_y = torch.from_numpy(np.array(train["labels"])).long()
# convert the test features to torch tensor
test_x = torch.from_numpy(test["data"]).float()
# reshape the test features to match the network input shape
test_x = test_x.view(-1, 3, 32, 32)
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


# start the training loop
for epoch in range(epochs):
    # initialize the running loss and accuracy for train data
    train_loss = 0.0
    train_acc = 0.0
    # loop through the train data
    for i, (inputs, labels) in enumerate(train_loader):
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
        # print the statistics every 200 batches
        if (i + 1) % 200 == 0:
            print(
                f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {train_loss / (i + 1):.4f}, Accuracy: {train_acc / (i + 1):.4f}")

    # initialize the running loss and accuracy for test data
    test_loss = 0.0
    test_acc = 0.0
    # loop through the test data
    for inputs, labels in test_loader:
        # forward propagation
        outputs = net(inputs)
        # calculate the loss
        loss = criterion(outputs, labels)
        # accumulate the running loss and accuracy
        test_loss += loss.item()
        test_acc += accuracy(outputs, labels)

    # print the statistics for each epoch
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_acc / len(train_loader):.4f}, Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_acc / len(test_loader):.4f}")

# save the trained network model
torch.save(net.state_dict(), "lenet5_model.pth")
