import torch
import torch.nn as nn
import torch.utils.data.dataset as dataset
import torchvision.transforms as transforms
import torchvision

#hyper parameters
num_epochs = 5
batch_size = 100
learning_rate =  0.001

#Image augmentation
transform = transforms.Compose([transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()])

train_dataset = torchvision.datasets.CIFAR10(root='C://', train=True, transform=transform, download=False)
test_data_set =torchvision.datasets.CIFAR10(root='C://', train=False, transform=transforms.ToTensor())

train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data_set, batch_size=batch_size, shuffle=False)

#resNet
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class resBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,downsampling=None):
        super(resBlock,self).__init__()
        self.conv1 = conv3x3(in_channels,out_channels,stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels,out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsampling = downsampling

    def forward(self,x):
        residual = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampling:
            residual = self.downsampling(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, resBlock,layers, num_classes=10):
        """ layers [2,2,2] """
        super(ResNet,self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3,16)
        self.bn=nn.BatchNorm2d(16)
        self.relu=nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(resBlock,16,layers[0])
        self.layer2 = self.make_layer(resBlock,32,layers[1],stride=2)
        self.layer3 = self.make_layer(resBlock,64,layers[2],stride=2)
        self.avg_pool =  nn.AvgPool2d(8)
        self.fc = nn.Linear(64,num_classes)

    def make_layer(self,resBlock,out_channels,res_blocks,stride=1):
        downsampling = None
        if (stride!=1) or (self.in_channels != out_channels):
            downsampling = nn.Sequential(conv3x3(self.in_channels,out_channels,stride=stride),nn.BatchNorm2d(out_channels))
        residual_blocks = []
        residual_blocks.append(resBlock(self.in_channels,out_channels,stride,downsampling))
        self.in_channels = out_channels
        for i in range(1, res_blocks):
            residual_blocks.append(resBlock(out_channels ,out_channels))
        return nn.Sequential(*residual_blocks)


    def forward(self,x):
        # x is the input
        out=self.conv(x)
        out=self.bn(out)
        out=self.relu(out)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.avg_pool(out)
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out


model = ResNet(resBlock, [2,2,2])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.train()
decay=0
for epoch in range(num_epochs):
    #decay learning rate every 20 epochs
    if (epoch+1)%20 == 0:
        decay+=1
        optimizer.param_groups[0]['lr'] = learning_rate * (0.5*decay)
        print(f"The new learning rate is {optimizer.param_groups[0]['lr']}")

    for i, (image,labels) in enumerate(train_data_loader):
        outputs = model(image)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1)%100==0:
            print(f"epoch {epoch+1}, step {i+1}, loss {loss.item()}")




model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in test_data_loader:
        outputs = model(images)
        _,predicted = torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    print(f"Accuracy {100*correct/total}")