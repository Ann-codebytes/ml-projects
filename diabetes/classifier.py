import numpy as np
import sklearn.preprocessing
import torch as tf
import torch.nn as nn
import pandas as pd
import torch.utils.data
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# load the data
data = pd.read_csv("diabetes.csv")
data.info()

x = data.drop("Class", axis=1).values
y = data["Class"].map({"negative": 0, "positive": 1})

# convert y from class to int value
y = y.values.astype("float64")

print(x)
print(y)

# normalize the data

x = StandardScaler().fit_transform(x)
print(x)


# convert the data to tensor
x = tf.tensor(x)
y = tf.tensor(y)

y = y.unsqueeze(1)

print(y.shape)
print(x.shape)


# create and load custom dataset
class DiabetesDataset(Dataset):
    def __init__(self):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


dataset = DiabetesDataset()

data_loader_db = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
for x, y in data_loader_db:
    print(x.shape, y.shape)


# build the neural network
class Model(nn.Module):
    def __init__(self, input_features, output_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_features, 5)
        self.fc2 = nn.Linear(5, 4)
        self.fc3 = nn.Linear(4, 3)
        self.fc4 = nn.Linear(3, 1)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out


# create the network
net = Model(7, 1)
# BCE Loss
criterion = torch.nn.BCELoss(size_average=True)
# Optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


# training the network
epochs = 200
for epoch in range(epochs):
    for inputs, labels in data_loader_db:
        inputs = inputs.float()
        labels = labels.float()
        # Forward propagation
        outputs = net(inputs)
        # loss calculation
        loss = criterion(outputs, labels)
        # clear the gradient buffer
        optimizer.zero_grad()
        # backward propagation
        loss.backward()
        # update weights
        optimizer.step()
    # accuracy calculation
    outputs = (outputs > 0.5).float()
    accuracy = (outputs == labels).float().mean()
    print(f"Epoch {epoch} - Loss: {loss.item()} - Accuracy: {accuracy.item()}")
