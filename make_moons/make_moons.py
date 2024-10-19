import torch as tf
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sklearn.datasets

(x,y) = sklearn.datasets.make_moons(200, noise=0.20)
print(x.shape)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.coolwarm)
plt.show()

x=tf.tensor(x).float()
y=tf.tensor(y).float()
y=y.unsqueeze(1)
print(x.shape)
print(y.shape)


class FeedForward(nn.Module):
    def __init__(self,input_features, hidden_layer, output_features):
        super(FeedForward,self).__init__()
        self.fc1 = nn.Linear(input_features,hidden_layer)
        self.fc2 = nn.Linear(hidden_layer,output_features)
        self.ReLU=nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

    def forward(self,x):
        out = self.fc1(x)
        out = self.ReLU(out)
        out=self.fc2(out)
        out=self.Sigmoid(out)
        return out


model = FeedForward(2,3,1)
loss = nn.BCELoss()
optimizer=tf.optim.SGD(model.parameters(),lr=0.02)

epochs = 1000
for epoch in range(epochs):
    output=model(x)
    optimizer.zero_grad()
    loss_val = loss(output,y)
    loss_val.backward()
    optimizer.step()
    prediction = (output>=0.5)
    accuracy = (prediction==y).float().mean()
    print(f"epoch{epoch}, loss{loss_val}, accuracy{accuracy}")
    if epoch%10 == 0:
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=prediction.data.numpy().flatten(),s=40,alpha=0.5)
        plt.text(3,1,f"Accuracy {accuracy}",fontdict = {'size':20,'color':'red'})
        plt.pause(0.1)
plt.show()
pt,fig = plt.subplots(2)
fig[0].scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.coolwarm)
fig[1].scatter(x[:,0],x[:,1],c=prediction,cmap=plt.cm.coolwarm)
plt.show()