import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.autograd import variable
from torchvision import datasets, models , transforms
import os
import numpy as np

# Data augmentation and normalization for training
# All training images are of size 224x224
data_transforms = {
    'train':transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'val':transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

data_dir = 'hymenoptera_data'

image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train', 'val'] }
dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x],batch_size =4 ,shuffle = True) for x in['train','val']}
data_sizes = {x:len(image_datasets[x])for x in ['train','val']}

class_names = image_datasets['train'].classes

print(f"Class names : {class_names}")
print(f"{len(dataloaders['train'])} batches for train")
print(f"{len(dataloaders['val'])} batches for train")
print(f"Training data size {data_sizes['train']}")
print(f"Val data size {data_sizes['val']}")

#Load the ResNet for the model which was previously trained
model_conv = torchvision.models.resnet18(pretrained=True)

#Freeze all layers except the final layer, we actually need the last layer modified but for that first freeze everything
for param in model_conv.parameters():
    param.required_grad=False

#modify the last layer to just classify between ant and bees so we just have two possible classes
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs,2)

#Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_conv.parameters(),lr=0.001,momentum=0.9)
#learning rate scheduler to delcay the LR by 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)

#train the model
num_epochs = 25
for epoch in range(num_epochs):
    exp_lr_scheduler.step()
    correct =  0
    for images, labels in dataloaders['train']:
        images = torch.tensor(images)
        labels = torch.tensor(labels)
        optimizer.zero_grad()
        outputs = model_conv(images)
        _,pred = torch.max(outputs,1)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        correct += torch.sum(pred == labels.data)
    epoch_acc = correct.double()/data_sizes['train']

    train_accuracy = 100* correct/data_sizes['train']
    print("Epoch: {} Training accuracy: {:.2f}".format(epoch,train_accuracy))

#test the model
model_conv.eval()
with torch.no_grad():
    correct =  0
    total =  0
    for images, labels in dataloaders['val']:
        images = torch.tensor(images)
        labels = torch.tensor(labels)
        outputs = model_conv(images)
        _,pred = torch.max(outputs,1)
        total += labels.size(0)
        correct += torch.sum(pred == labels.data).item()
    val_accuracy = 100* correct/total
    print("Validation accuracy: {:.2f}".format(val_accuracy))