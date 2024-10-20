import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image(path, img_transforms, size=(300, 300)):
    image = Image.open(path)
    image = image.resize(size, Image.LANCZOS)
    image = img_transforms(image)
    # to add the batch size - as our batch size is 1, in torch the first dimension has to be batch size
    image = image.unsqueeze(0)
    return image.to(device)


def get_gram(m):
    """m is of shape batch size that is always 1,C, H, W, channels , height and width"""
    _, c, h, w = m.size()
    m = m.view(c, h * w)
    gram = torch.mm(m, m.t())
    return gram


def denormalize_image(input):
    input = input.numpy().transpose(
        (1, 2, 0)
    )  # size of this input will be cxhxw ---> h,w,c
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = input * std + mean
    input = np.clip(input, 0, 1)
    return input


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # we are picking after each of the following layers, typically after ReLU as we need the non linearity introduced by ReLU
        self.selected_layers = [3, 8, 15, 22]
        self.vgg = models.vgg16(pretrained=True).features

    def forward(self, x):
        features = []
        for layer_number, layer in self.vgg._modules.items():
            x = layer(x)
            if int(layer_number) in self.selected_layers:
                features.append(x)
        return features


# img_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.485),std=(0.229))])
img_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

content_image = get_image("pic1.jpeg", img_transform)
style_image = get_image("style.jpg", img_transform)
# initialize the generated image to content, then alter based on the style, so set required grad to True
generated_img = content_image.clone()
generated_img.requires_grad = True


optimizer = torch.optim.Adam([generated_img], lr=0.003, betas=[0.5, 0.999])
encoder = FeatureExtractor().to(device)
for p in encoder.parameters():
    p.requires_grad = False

# or use encoder.eval() and that will prevent calculating grads
content_weight = 1
style_weight = 100
for epoch in range(500):
    content_features = encoder(content_image)
    style_features = encoder(style_image)
    generated_features = encoder(generated_img)
    # last selected layer's mse
    content_loss_mse_val = torch.mean(
        (content_features[-1] - generated_features[-1]) ** 2
    )
    style_loss = 0
    for s, g in zip(style_features, generated_features):
        _, c, h, w = g.size()
        # get the gram matrix for style
        gram_s = get_gram(s)
        gram_g = get_gram(g)
        style_loss += torch.mean((gram_s - gram_g) ** 2) / (c * h * w)

    total_loss = content_weight * content_loss_mse_val + style_weight * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch} Content loss: {content_loss_mse_val.item()} Style loss: {style_loss.item()} Total loss: {total_loss.item()}"
        )
image = generated_img.detach().squeeze(0)
image = denormalize_image(image)
plt.imshow(image)
plt.axis("off")
plt.savefig("style_added_generated_image.jpg", bbox_inches="tight")
plt.show()
