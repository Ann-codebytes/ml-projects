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


# create custom dataset
class DBDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


dataset = DBDataset(x, y)

data_loader_db = torch.utils.data.DataLoader(dataset, batch_size=24, shuffle=True)

for x, y in data_loader_db:
    print(x.shape, y.shape)


# define the Neural Network
class Model(nn.Module):
    def __init__(self, input_features, output_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_features, 7)
        self.fc2 = nn.Linear(7, 5)
        self.fc3 = nn.Linear(5, 3)
        self.fc4 = nn.Linear(3, output_features)
        self.sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.ReLU(out)
        out = self.fc2(out)
        out = self.ReLU(out)
        out = self.fc3(out)
        out = self.ReLU(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out


# create network
net = Model(7, 1)
loss = nn.BCELoss(size_average=True)
optimizer = tf.optim.Adam(net.parameters(), lr=0.01)

# training the network
epochs = 200
for epoch in range(epochs):
    for inputs, labels in data_loader_db:
        inputs = inputs.float()
        labels = labels.float()
        outputs = net(inputs)
        optimizer.zero_grad()
        loss_val = loss(outputs, labels)
        loss_val.backward()
        optimizer.step()

    # accuracy
    outputs = (outputs >= 0.5).float()
    accuracy = (outputs == labels).float().mean()
    print(f"Epoch {epoch} Loss {loss_val} accuracy{accuracy}")
