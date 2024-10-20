import numpy as np
import torch
import torch.nn as nn
import torch.utils.data.dataset as dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("airbnb.csv")
data.info()

data = data.dropna()
data.info()
print(data.loc[0, "rating"])

# Predict the price based on rating, price, country, bathroomds, beds, guests, bedrooms,studios

data_in = data[
    ["rating", "price", "country", "bathrooms", "beds", "guests", "bedrooms", "studios"]
]
data_out = data["price"]
data_in = data_in.drop("price", axis=1)

data_in.info()
data_out.info()
print(data_in.shape)
print(data_out.shape)

rating_vals = data_in["rating"].unique()
rating_map = {}
for index in range(len(rating_vals)):
    rating_map[rating_vals[index]] = index
data_in["rating"] = data_in["rating"].map(rating_map)

data_in["rating"] = data_in["rating"].astype(float)
data_in.info()

country_vals = data_in["country"].unique()
country_map = {}
for index in range(len(country_vals)):
    country_map[country_vals[index]] = index
data_in["country"] = data_in["country"].map(country_map)
data_in["country"] = data_in["country"].astype(int)

data_in.info()
data_in_original = data_in
data_out_original = data_out
data_in_original = StandardScaler().fit_transform(data_in_original)

data_in, data_test_in, data_out, data_test_out = train_test_split(
    data_in_original, data_out_original, test_size=0.3, random_state=0
)

data_in = torch.Tensor(data_in)
data_out = torch.Tensor(data_out)


class AirBnBDataset(dataset.Dataset):
    def __init__(self):
        self.x = data_in
        self.y = data_out

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class testModel(nn.Module):
    def __init__(self, input_layer, hidden_layer, output_layer):
        super(testModel, self).__init__()
        self.fc1 = nn.Linear(input_layer, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, hidden_layer)
        self.fc4 = nn.Linear(hidden_layer, output_layer)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


air_data_set = AirBnBDataset()
data_loader_air = torch.utils.data.DataLoader(air_data_set, shuffle=True, batch_size=32)
input_size = 7
hidden_size = 15
output_size = 1
airNet = testModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(airNet.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    for input, value in data_loader_air:
        output_val = airNet(input.float())
        loss = criterion(output_val, value.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch:", epoch, "Loss:", loss.item())

# To test:
data_in = torch.Tensor(data_test_in)
data_out = torch.Tensor(np.array(data_test_out))

air_data_set_test = AirBnBDataset()
air_data_set_test_loader = torch.utils.data.DataLoader(
    air_data_set_test, shuffle=True, batch_size=32
)

for input, value in air_data_set_test_loader:
    output_val = airNet(input)
    print(f"Output {output_val} and Actual{value}")
