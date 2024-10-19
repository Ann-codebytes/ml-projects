#Kaggle data set reference: https://www.kaggle.com/datasets/emirhanai/social-media-usage-and-emotional-well-being/code

import torch as tf
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def clean_data(file_name):
    #load the data
    data = pd.read_csv(file_name)
    data.info()

    #Clean the data
    data = data.dropna()
    data["Gender"]=data["Gender"].map({"Male":0,"Female":1,"Non-binary":2})
    map_platforms = {}
    for index in range(0,len(data["Platform"].unique())):
        map_platforms[data["Platform"].unique()[index]] = index
    data["Platform"]=data["Platform"].map(map_platforms)
    data['Platform'] = data['Platform'].astype(float)
    map_dominant_emotion = {}
    for index in range(0,len(data["Dominant_Emotion"].unique())):
        map_dominant_emotion[data["Dominant_Emotion"].unique()[index]] = index
    data["Dominant_Emotion"]=data["Dominant_Emotion"].map(map_dominant_emotion)
    data['Dominant_Emotion'] = data['Dominant_Emotion'].astype(float)
    map_gender = {}
    for index in range(0,len(data["Gender"].unique())):
        map_gender[data["Gender"].unique()[index]] = index
    data["Gender"]=data["Gender"].map(map_gender)
    data['Gender'] = data['Gender'].astype(float)
    data['User_ID'] = data['User_ID'].astype(float)
    #Age has invalid values so drop the field
    data = data.drop("Age",axis=1)
    data.info()
    x= data.drop("Dominant_Emotion",axis=1).values
    y= data["Dominant_Emotion"].values

    data.info()
    return x, y

x, y = clean_data('train.csv')
x_val, y_val = x,y
#normalize the data
x=StandardScaler().fit_transform(x)
x=tf.tensor(x)
y=tf.tensor(y)

print(x.shape)
print(y.shape)

#create and load the dataset
class SocEmDataSet(Dataset):
    def __init__(self):
        self.x=x
        self.y=y

    def __getitem__(self,index):
        return self.x[index],self.y[index]

    def __len__(self):
        return len(self.x)


#build network
class Net(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(Net,self).__init__()
        self.fc1=nn.Linear(input_size, hidden_size)
        self.fc2=nn.Linear(hidden_size,hidden_size)
        self.fc3=nn.Linear(hidden_size,hidden_size)
        self.fc4=nn.Linear(hidden_size,hidden_size)
        self.fc5=nn.Linear(hidden_size,output_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu =nn.ReLU()
        self.softmax =nn.Softmax()

    def forward(self,x):
        out=self.fc1(x)
        out=self.relu(out)
        out=self.fc2(out)
        out=self.relu(out)
        out=self.fc3(out)
        out=self.relu(out)
        out=self.fc4(out)
        out=self.relu(out)
        out=self.fc5(out)
        out=self.softmax(out)
        return out
#whatever you put in x and y is being picked by the Dataset as can be seen in init, so we just put the right value in x and y
#that is why we redefine x and y just before defining validation set and test set as well
dataset = SocEmDataSet()
data_loader_db = tf.utils.data.DataLoader(dataset, batch_size=40, shuffle=True)
for (x,y) in data_loader_db:
    print(x.shape,y.shape)

x_val=x_val
y_val=y_val
x_val=StandardScaler().fit_transform(x_val)
x_val=tf.tensor(x_val)
y_val=tf.tensor(y_val)
x=x_val
y=y_val
dataset = SocEmDataSet()
data_loader_db_val = tf.utils.data.DataLoader(dataset, batch_size=40, shuffle=True)
for (x,y) in data_loader_db_val:
    print(x.shape,y.shape)

x_test,y_test=clean_data('test.csv')
x_test=StandardScaler().fit_transform(x_test)
x_test=tf.tensor(x_test)
y_test=tf.tensor(y_test)
x=x_test
y=y_test
dataset = SocEmDataSet()
data_loader_db_test = tf.utils.data.DataLoader(dataset, batch_size=150, shuffle=True)
for (x_test,y_test) in data_loader_db_test:
    print(x_test.shape,y_test.shape)

#create the Model
#there are 8 input features
#130 neurons in hidden layer, that we just decided on random
#there are size possible dominant emotion classes, so that is the output layer
input_size = 8
output_size=6
hidden = 130
socNet = Net(input_size,hidden,output_size)
criterion = nn.CrossEntropyLoss()
optimizer = tf.optim.Adam(socNet.parameters(),lr=0.0006)

#train the model
epochs = 300
accuracy=0
for epoch in range(epochs):
    for inputs, labels in data_loader_db:
        for inputs_val, labels_val in data_loader_db_val:
            inputs = inputs.float()
            labels = labels.long()
            outputs = socNet(inputs)
            predicted = tf.max(outputs, 1)[1]
            train_loss = criterion(outputs,labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            accuracy = (predicted == labels).float().mean()
            print(f"TRAIN Epoch {epoch}, Loss {train_loss}, Accuracy {accuracy}")

            inputs_val = inputs_val.float()
            labels_val = labels_val.long()
            outputs_val = socNet(inputs_val)
            predicted_val = tf.max(outputs_val, 1)[1]
            loss = criterion(outputs_val,labels_val)
            accuracy = (predicted_val == labels_val).float().mean()
            print(f"VAL Epoch {epoch}, Loss {loss}, Accuracy {accuracy}")
            if loss > train_loss:
                #no need to continue if validation loss is greater than training loss
                break

print(f" Parameters {list(socNet.named_parameters())}")

#test the model
accuracy=0
for inputs_test, labels_test in data_loader_db_test:
    inputs = inputs_test.float()
    labels = labels_test.long()
    print(f" NET Parameters {list(socNet.named_parameters())}")
    outputs = socNet(inputs)
    predicted = tf.max(outputs, 1)[1]
    accuracy = (predicted == labels).float().mean()
print(f"Testing Accuracy {accuracy}")

conf_matrix = confusion_matrix(labels,predicted)
print(conf_matrix)