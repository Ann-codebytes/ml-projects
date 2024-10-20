import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from imageio import imread
import cv2


# Saving model and weights


def process_image(image):
    img = imread(image)
    img = cv2.resize(img, (256, 256))  # 256x256x3
    img = img.transpose(2, 0, 1)  # channels first
    img = img / 255  # normalize
    img = torch.FloatTensor(img)
    # these are means and std of each channel - mean=[0.485,0.456, 0.406],std=[0.229, 0.224, 0.225]
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([normalize])
    img = transform(img)  # (3,256,256)
    return img


# define the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # input channels is the output of the previous layer that was 6 channels
        self.conv3 = nn.Conv2d(6, 12, 5)
        # flatten the feature map, input is the number of pixesl in the feature map from conv3
        self.fc1 = nn.Linear(12 * 61 * 61, 120)
        # output of previous layer is input to the next layer
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = F.relu(
            self.conv1(x)
        )  # output size  =[ (256-5 +(2 *0)) / stride] + 1, that is 256 is pixels, 5 is the kernel size, 2* padding of zero divided by stride that is 1 => 252x252
        x = self.pool(x)  # max pool with 2 so we reduce size by 2, so we get 126x126
        x = F.relu(
            self.conv3(x)
        )  # output size in this case is [(126 - 5 + 2(0))/1]+1=122x22
        x = self.pool(x)  # output size = 122/2 = 61x61
        x = x.view(-1, 12 * 61 * 61)  # (1,44652)
        x = F.relu(self.fc1(x))  # (1,120)
        x = self.fc2(x)  # (1,10)
        return x


# Initialize the model
model = CNN()
print(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
image = process_image("test.jpg")
image = image.unsqueeze(0)  # batch dimension, only one image so it will be a 1

# Print model's state dict
print("Model's state dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state dict
print("Optimizer's state dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

# saving and loading a model

# Save the model
torch.save(model.state_dict(), "model.pth.tar")

# Now you want to load and test
# the model class definition is still needed even to use a saved model
model = CNN()
model.load_state_dict(torch.load("model.pth.tar", weights_only=True))
model.eval()  # we want to use dropout and batch normalization to be used only in training so they must be in eval model they act differently in testing
# in this definition the droput and batch layer are missing but add this for completeness


# saving and loading a general checkpit for inference and/or resuming training
checkpoint = {
    "epoch": 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": 0.2,
}
torch.save(checkpoint, "model.pth.tar")

model = CNN()
# this will not have effect as we are loading the already stated checkpoint before the print here
optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.23423)
checkpoint = torch.load("model.pth.tar", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]

# If testing
model.eval()
# If resuming training
model.train()

print(epoch)
print(optimizer)
