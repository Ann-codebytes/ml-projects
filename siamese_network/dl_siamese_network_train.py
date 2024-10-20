import pandas as pd
import os
import cv2
import torchvision.models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models
from torchvision import transforms
import torch.nn as nn
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=os.path.join(os.getcwd(), "trainlog.log"),
    filemode="w",
)

learning_rate_training = 0.005
num_epochs = 10


def list_all_image_pairs(base_dir):
    families_directory = os.path.join(base_dir, "train")
    families = os.listdir(families_directory)

    x = []
    y = []
    z = {}
    w = {}
    ind_counter = 0

    for family in families:
        z[int(family[1:])] = {}
        last_counter = ind_counter
        for relative in os.listdir(os.path.join(families_directory, family)):
            relative_dir = os.path.join(families_directory, family)
            z[int(family[1:])][int(relative[3:])] = []
            pic_counter = 0
            for pic in os.listdir(os.path.join(relative_dir, relative)):
                pic_dic = os.path.join(relative_dir, relative)
                # resize_img = cv2.resize(np.asarray(Image.open(os.path.join(pic_dic, pic)).convert('L')),(64,64))
                resize_img = np.asarray(
                    Image.open(os.path.join(pic_dic, pic)).convert("L")
                )
                x.append(resize_img)
                # y has relatives of x, each relative is appended separate
                y.append([int(family[1:]), int(relative[3:]), pic_counter])
                w[ind_counter] = []
                # indicate which indices are related
                for ind_pic in range(last_counter, ind_counter):
                    w[ind_pic].append(ind_counter)
                for ind_pic in range(last_counter, ind_counter + 1):
                    w[ind_counter].append(ind_pic)
                pic_counter += 1
                ind_counter += 1

    related = pd.read_csv(os.path.join(base_dir, "train_relationships.csv"))
    # make a connection between the related indices
    for ind in range(0, len(related)):  # len(related)
        try:
            z[int(related.iloc[ind]["p1"].split("/")[0][1:])][
                int(related.iloc[ind]["p1"].split("/")[1][3:])
            ].append(int(related.iloc[ind]["p2"].split("/")[1][3:]))
            z[int(related.iloc[ind]["p1"].split("/")[0][1:])][
                int(related.iloc[ind]["p2"].split("/")[1][3:])
            ].append(int(related.iloc[ind]["p1"].split("/")[1][3:]))
        except:
            print("ind ", ind, " Index error")
    # print(x)
    # print(y)
    # print(w)
    # print(z)
    x = np.asarray(x) / 255.0
    return x, y, w, z


def get_images_for_training(x, y, w, z, size_list=64):
    rand_indexes_1 = []
    train_x = []
    train_y = np.zeros(size_list)
    train_x.append(
        np.zeros((size_list, 224, 224, 1))
    )  # final size is (2, batch_size, 224, 224, 1)
    train_x.append(np.zeros((size_list, 224, 224, 1)))
    ran_values = np.random.randint(
        0, 2, (1, size_list)
    )  # Random value which determines whether to chose from relatives or not , I need both categories for training
    rand_indexes_0 = np.random.choice(
        range(0, len(x)), size_list, replace=False
    )  # Random Indexes for choosing the first half of images
    for ind_batchy in range(
        0, size_list
    ):  # Creating the 2nd half of pictures either from relatives or non-relatives
        if ran_values[0][ind_batchy] > 0.5:
            rand_val = np.random.choice(w[rand_indexes_0[ind_batchy]], 1, replace=False)
            rand_indexes_1.append(int(rand_val[0]))
        else:
            rand_val = np.random.choice(range(0, len(x)), 1, replace=False)
            rand_indexes_1.append(int(rand_val[0]))
    rand_indexes_1 = np.asarray(rand_indexes_1)
    train_x[0] = x[rand_indexes_0]
    train_x[1] = x[rand_indexes_1]
    for ind_batch in range(0, size_list):  # Creating labels for the batches of pictures
        if y[rand_indexes_0[ind_batch]][0] == y[int(rand_indexes_1[ind_batch])][0]:
            if (
                y[rand_indexes_0[ind_batch]][1]
                in z[y[int(rand_indexes_1[ind_batch])][0]][
                    y[int(rand_indexes_1[ind_batch])][1]
                ]
            ) or (
                y[rand_indexes_0[ind_batch]][1] == y[int(rand_indexes_1[ind_batch])][1]
            ):
                train_y[ind_batch] = 1
            else:
                train_y[ind_batch] = 0
        else:
            train_y[ind_batch] = 0
    return (train_x, train_y)


def print_vals(train_x, train_y):
    print(len(train_x))
    print(len(train_y))
    # print shape of train_x and train_y
    print(train_x[0].shape)
    print(train_x[1].shape)
    fiwd_data = FIWDataset(train_x, train_y)
    # print(len(fiwd_data))
    print(
        fiwd_data.__getitem__(0)
    )  # this returns, image 1, image 2 and label that is expected
    plt.imshow(fiwd_data.__getitem__(0)[0].numpy())
    plt.show()
    plt.imshow(fiwd_data.__getitem__(0)[1].numpy())
    plt.show()
    print(fiwd_data.__getitem__(0)[1].shape)


class FIWDataset:
    def __init__(self, train_x, train_y):
        self.x_img1 = train_x[0]
        self.x_img2 = train_x[1]
        self.y = train_y

    def __getitem__(self, index):
        img1 = torch.tensor(self.x_img1[index]).unsqueeze(0)
        img2 = torch.tensor(self.x_img2[index]).unsqueeze(0)
        y_val = torch.tensor(self.y[index])
        return img1, img2, y_val

    def __len__(self):
        return len(self.y)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Load the ResNet for the model which was previously trained
        self.model_conv = torchvision.models.resnet50(pretrained=True)

        # Freeze all layers except the final layer, I actually need the last layer modified but for that first freeze everything
        for param in self.model_conv.parameters():
            param.required_grad = False
        self.model_conv.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        num_ftrs = self.model_conv.fc.in_features
        self.model_conv = torch.nn.Sequential(*(list(self.model_conv.children())[:-1]))

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        # initialize the weights
        # self.model_conv.apply(self.init_weights)
        # self.fc.apply(self.init_weights)
        logging.info(f"Model : {self.model_conv}")

    def forward_once(self, x):
        output = self.model_conv(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, img1, img2):
        # get two images' features
        output1 = self.forward_once(img1)
        output2 = self.forward_once(img2)

        return output1, output2

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss


if __name__ == "__main__":
    # list all images
    x, y, w, z = list_all_image_pairs(r"C:\\")
    train_x, train_y = get_images_for_training(x, y, w, z, 1000)
    # print size of train_x and train_y
    # print_vals(train_x,train_y)
    dataset = FIWDataset(train_x, train_y)
    #
    # #dataset
    data_loader_db = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, drop_last=True
    )
    for img1, img2, y in data_loader_db:
        print(img1.shape, img2.shape, y.shape)

    # train
    net = SiameseNetwork()
    logging.info(f"Model {net.model_conv}")
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate_training)
    epoch_loss = {}
    for epoch in range(num_epochs):
        for img1, img2, y in data_loader_db:
            optimizer.zero_grad()
            output1, output2 = net(
                torch.tensor(img1.float()), torch.tensor(img2.float())
            )
            loss = criterion(output1, output2, y)
            loss.backward()
            optimizer.step()
        logging.info(f"Epoch: {epoch} Loss: {loss.item()}")
        epoch_loss[epoch] = loss.item()
    logging.info(f"Epoch Loss {epoch_loss}")
    torch.save(net.state_dict(), "model.pth.tar")
