import os

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#input parameters
input_size = 784
hidden_size = 400
out_size = 10
epochs = 10
batch_size = 100
learning_rate = 0.001

print(os.getcwd())
training_data = datasets.MNIST(root='C://', train=True, download=False, transform=transforms.ToTensor())
testing_data = datasets.MNIST(root='C://', train=False, download=False, transform=transforms.ToTensor())

training_data_loader = torch.utils.data.dataloader.DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
testing_data_loader = torch.utils.data.dataloader.DataLoader(dataset = testing_data,batch_size = batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self,input_size,hidden_size,out_size):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,out_size)
        self.relu =  nn.ReLU()

    def forward(self,input):
        out = self.fc1(input)
        out = self.relu(out)
        out=self.fc2(out)
        out=self.relu(out)
        return out

net = Net(input_size,hidden_size,out_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

for epoch in range(epochs):
    for i, data in training_data_loader:
        images , labels = i, data
        images = images.view(-1,28*28)
        labels = labels
        output = net(images)
        loss = criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predicted = torch.max(output,1)[1]
        accuracy = (predicted == labels).sum().item()/batch_size
        print(f"Epoch {epoch}, Loss {loss}, Accuracy {accuracy}")

accuracy =  0
with torch.no_grad():
    for i, data in enumerate(testing_data_loader):
        images, labels = data
        images = images.view(-1,28*28)
        labels = labels
        output = net.forward(images)
        predicted = torch.max(output,1)[1]
        accuracy  += (predicted == labels).sum().item()/batch_size
print(f"Testing Accuracy {accuracy/len(testing_data_loader)}")
