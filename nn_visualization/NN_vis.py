import torch
import torch.nn as nn
import torch.autograd as Variable
from torchvision import models
from torchvision import transforms,utils
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
import json
#declare transforms
transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406], std= [0.229,0.224,0.225])])

image = Image.open('dog.jpg')
plt.imshow(image)
plt.show()

vgg = models.vgg16(pretrained=True)

print(vgg)

#apply transforms
image=transform(image)

print(image.shape)
image = image.unsqueeze(0)
print(image.shape)
output = vgg(image)
print(output.shape)
output=output.squeeze(0)
print(output.shape)

labels = json.load(open('imagenet_class_index.json'))
print(labels)
index=output.max(0)
print(index)

index=str(index[1].item())
label = labels[index][1]
print(label)

#access the features now
module_list = list(vgg.features.modules())
print(vgg.features)
print(module_list[0])
print(module_list[1])
print(module_list[2])


#Visualize the feature maps

outputs = []
names = []
for layer in module_list[1:]:
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))

for feature_maps in outputs:
    print(feature_maps.shape)

processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0) #remove 1
    #sum 3d to 2d, sum elements of all channels and sum them up
    gray_scale = torch.sum(feature_map,0)
    #normalized visualization, divide by 64,128,256 etc
    gray_scale = gray_scale/feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())

fig = plt.figure(figsize = (30,50))

for index in range(len(processed)):
    a = fig.add_subplot(8,4,index+1)
    imgplot = plt.imshow(processed[index])
    plt.axis('off')
    a.set_title(names[index].split('(')[0],fontsize=30)

plt.savefig('feature_maps.jpg', bbox_inches='tight')
