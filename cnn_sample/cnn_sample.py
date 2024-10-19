import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

mean_gray =  0.1397
stddev_gray = 0.3081

#data load
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((mean_gray,),(stddev_gray,))])
training_data = datasets.MNIST(root='C://', train=True, download=False, transform=transform)
testing_data = datasets.MNIST(root='C://', train=False, download=False, transform=transform)

# random_img = training_data[20][0].numpy()*stddev_gray + mean_gray
# plt.imshow(random_img.reshape(28,28), cmap='grey')
# plt.xlabel(training_data[20][1])
# plt.show()

# data preprocessing
batch_size = 100
training_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
testing_loader = torch.utils.data.DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True)

print(len(training_data))
print(len(testing_data))

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #same padding means input size is same as output size
        #same padding= (kernel_size - 1)/2 -> (3-1)/2=1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        #output size for each feature map
        #(input_size - filtersize + 2*padding)/stride + 1 = 28-3+2*1/1+1=28
        #output_size = 28
        #batch normalization
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        #output size for each feature map 28/2=14
        #in channels is how many feature maps you got from previous layer
        #out channels is the number of filters you are applying now, so in this case we are applying 32 filters
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
        #output size for each feature map
        #same padding= (kernel_size - 1)/2 -> (5-1)/2=2
        #(input_size - filtersize + 2*padding)/stride + 1 = 14-5+2*2/1+1=14
        #output_size = 14
        #batch normalization
        self.batchnorm2 = nn.BatchNorm2d(32)

        #Now the feature map size is 7x7 and you have 32 feature maps
        # so flattten you have 7x7x32 that is 1568
        #pass that to the first fully connected layer, we just choose a 600 as output features here
        self.fc1=nn.Linear(1568,600)
        #hidden layer has 600 and then we have 10 classes so output layer has 10
        #drop out layer
        self.droput = nn.Dropout(p=0.5)
        self.fc2=nn.Linear(600,10)

    def forward(self,x):
            out=self.cnn1(x)
            out=self.batchnorm1(out)
            out=self.relu(out)
            out=self.maxpool(out)
            out=self.cnn2(out)
            out=self.relu(out)
            out=self.maxpool(out)


            #flatten, batch size , 1568 works but -1 will make pytorch figure out itself. Python figures out if we put a -1
            out=out.view(-1,1568)
            out=self.fc1(out)
            out=self.relu(out)
            out=self.droput(out)
            out=self.fc2(out)
            return out

model = CNN()
cuda = torch.cuda.is_available()
print(cuda)
#check if GPU is availble, if so move the model to the GPU
if cuda:
    model = model.cuda()

# for i,(inputs,labels)in enumerate(training_loader):
#     print(inputs.shape)
#     #if GPU is available move the inputs to the GPU
#     if cuda:
#         inputs = inputs.cuda()
#         labels = labels.cuda()
#     print(inputs.shape)

#when using softmax CrossEntropyLoss is the best when using sigmoid use BinaryCrossEntropyLoss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#To understand
# for i,(inputs,labels)in enumerate(training_loader):
#     print(inputs.shape)
#     #if GPU is available move the inputs to the GPU
#     if cuda:
#         inputs = inputs.cuda()
#         labels = labels.cuda()
#     output = model(inputs)
#     loss = criterion(output, labels)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     prediction = torch.max(output,1)[1]
#     accuracy = (prediction == labels).sum()
#     print(f"Accuracy {accuracy}")

epochs = 10
training_loss = []
testing_loss = []
training_accuracy = []
testing_accuracy = []
for epoch in range(epochs):
    model.train()
    iter_loss = 0.0
    correct = 0
    iteration =  0
    for i , (inputs, labels) in enumerate(training_loader):
        if cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        iter_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prediction = torch.max(outputs,1)[1]
        correct += (prediction == labels).sum().item()
        iteration+=1
        print(f"Epoch {epoch}, Batch {i}, Loss {loss.item()}")
    training_accuracy.append(correct/len(training_data))
    training_loss.append(iter_loss/iteration)

print(f"PRINT AS many batches as there are {iteration}")  # that should print a number close to 600
#loss is computed once per batch alone

#testing
model.eval()
#tells that we are doing testing (needed because we have the batch normalization and pooling layers)
iter_loss = 0.0
correct = 0
iteration =  0
for i , (inputs, labels) in enumerate(testing_loader):
    if cuda:
        inputs = inputs.cuda()
        labels = labels.cuda()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    iter_loss += loss.item()
    prediction = torch.max(outputs,1)[1]
    correct += (prediction == labels).sum().item()
    iteration +=1
    testing_accuracy.append(correct/len(testing_data))
    testing_loss.append(iter_loss/iteration)

print(iteration) #that should print a number close to 100

print(testing_accuracy)
fig = plt.figure(figsize = (10,10))
plt.plot(training_accuracy, label='Training accuracy')
plt.plot(testing_accuracy, label='Testing accuracy')
plt.legend()
plt.show()

print(testing_loss)
fig = plt.figure(figsize = (10,10))
plt.plot(training_loss, label='Training_loss')
plt.plot(testing_loss, label='Testing_loss')
plt.legend()
plt.show()

        









