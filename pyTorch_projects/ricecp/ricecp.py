# Reference : https://www.kaggle.com/datasets/thegoanpanda/rice-crop-diseases
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from PIL import Image as image
import torchvision
import glob
import os
import numpy as np
import pandas as pd

image_list = torch.tensor([])
y = []
adjust_size = 200
for directory in glob.glob("C:\\"):
    path = os.path.join("C:\\", directory)
    for pic in glob.glob(path + "/*.jpg"):
        im = image.open(pic).convert("L")
        im = im.resize((adjust_size, adjust_size), image.LANCZOS)
        image_tensor = torchvision.transforms.ToTensor()(im)
        image_tensor = image_tensor.view(-1, adjust_size * adjust_size)
        image_list = torch.cat((image_list, image_tensor), 0)
        y.append(directory.split("\\")[-1])
        # along with original image , do some data augmentation
        im = torchvision.transforms.RandomHorizontalFlip()(im)
        image_tensor = torchvision.transforms.ToTensor()(im)
        image_tensor = image_tensor.view(-1, adjust_size * adjust_size)
        image_list = torch.cat((image_list, image_tensor), 0)
        y.append(directory.split("\\")[-1])
        im = torchvision.transforms.RandomVerticalFlip()(im)
        image_tensor = torchvision.transforms.ToTensor()(im)
        image_tensor = image_tensor.view(-1, adjust_size * adjust_size)
        image_list = torch.cat((image_list, image_tensor), 0)
        y.append(directory.split("\\")[-1])
        im = torchvision.transforms.RandomHorizontalFlip()(im)
        image_tensor = torchvision.transforms.ToTensor()(im)
        image_tensor = image_tensor.view(-1, adjust_size * adjust_size)
        image_list = torch.cat((image_list, image_tensor), 0)
        y.append(directory.split("\\")[-1])
        im = torchvision.transforms.RandomVerticalFlip()(im)
        image_tensor = torchvision.transforms.ToTensor()(im)
        image_tensor = image_tensor.view(-1, adjust_size * adjust_size)
        image_list = torch.cat((image_list, image_tensor), 0)
        y.append(directory.split("\\")[-1])
        im = torchvision.transforms.RandomHorizontalFlip()(im)
        image_tensor = torchvision.transforms.ToTensor()(im)
        image_tensor = image_tensor.view(-1, adjust_size * adjust_size)
        image_list = torch.cat((image_list, image_tensor), 0)
        y.append(directory.split("\\")[-1])
        im = torchvision.transforms.RandomVerticalFlip()(im)
        image_tensor = torchvision.transforms.ToTensor()(im)
        image_tensor = image_tensor.view(-1, adjust_size * adjust_size)
        image_list = torch.cat((image_list, image_tensor), 0)
        y.append(directory.split("\\")[-1])


# network parameters:
input_size = adjust_size * adjust_size
hidden_size = 150
output_size = len(np.unique(np.array(y)))

print(image_list.size())
print(len(y))
map_y = {}
for index in range(0, len(np.unique(np.array(y)))):
    map_y[np.unique(np.array(y))[index]] = index

y_pd = pd.DataFrame(y, columns=["label"])
y_pd["label"] = y_pd["label"].map(map_y)
y = torch.Tensor(y_pd["label"].values)
print(image_list.shape)
print(y.shape)
train_x, test_candidate_x, train_y, test_candidate_y = train_test_split(
    image_list, y, test_size=0.2, random_state=0
)

test_x, val_x, test_y, val_y = train_test_split(
    test_candidate_x, test_candidate_y, test_size=0.1, random_state=0
)


# Define data set
class RiceDataset(Dataset):
    def __init__(self):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


x = train_x
# x=StandardScaler().fit_transform(x)
y = train_y
rice_data_train = RiceDataset()
data_loader_rice_train = torch.utils.data.DataLoader(
    rice_data_train, batch_size=32, shuffle=True
)

x = val_x
# x=StandardScaler().fit_transform(x)
y = val_y
rice_data_val = RiceDataset()
data_loader_rice_val = torch.utils.data.DataLoader(
    rice_data_val, batch_size=32, shuffle=True
)

x = test_x
# x=StandardScaler().fit_transform(x)
y = test_y
rice_data_test = RiceDataset()
data_loader_rice_test = torch.utils.data.DataLoader(
    rice_data_test, batch_size=700, shuffle=True
)


# create a network\
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        out = self.softmax(out)
        return out


# train the model
riceNet = Net(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(riceNet.parameters(), lr=0.0001)

epochs = 400
for epoch in range(epochs):
    for input, label in data_loader_rice_train:
        for input_val, label_val in data_loader_rice_val:
            input = input.float()
            label = label.long()
            output = riceNet(input)
            predicted = torch.max(output, 1)[1]
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy = (predicted == label).float().mean()
            print(f"TRAIN Epoch {epoch}, Loss {loss}, accuracy {accuracy}")

            input_val = input_val.float()
            label_val = label_val.long()
            output_val = riceNet(input_val)
            predicted_val = torch.max(output_val, 1)[1]
            loss_val = criterion(output_val, label_val)
            accuracy_val = (predicted_val == label_val).float().mean()
            print(f"VAL Epoch {epoch}, Loss {loss_val}, accuracy {accuracy_val}")


# test the model
for input_test, label_test in data_loader_rice_test:
    input_test = input_test.float()
    label_test = label_test.long()
    output_test = riceNet(input_test)
    predicted_test = torch.max(output_test, 1)[1]
    accuracy_test = (predicted_test == label_test).float().mean()
    print(f"TEST accuracy {accuracy_test}")
