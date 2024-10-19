import torch
import torch.nn as F
from proj_siamese_network_train import SiameseNetwork,list_all_image_pairs,get_images_for_training,FIWDataset
import logging



if __name__=="__main__":
    #list all images
    x,y,w,z= list_all_image_pairs(r'C:\\')
    train_x,train_y=get_images_for_training(x,y,w,z,1000)
    #print size of train_x and train_y
    # print_vals(train_x,train_y)
    dataset = FIWDataset(train_x,train_y)
    #
    # #dataset
    data_loader_db = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,drop_last=True)
    for (img1,img2,y) in data_loader_db:
         print(img1.shape,img2.shape,y.shape)

    # for testing
    model = SiameseNetwork()
    #load the saved model
    model.load_state_dict(torch.load('model.pth.tar', weights_only=True))
    model.eval()  # I want to use dropout and batch normalization to be used only in training
    #in testing model must be run with eval

    eucledian_distance = F.PairwiseDistance(p=2)
    count=0
    correct = 0
    threshold_euclidean_distance = 0.5
    subset_of_batches=len(data_loader_db)
    for (img1, img2, label) in data_loader_db:
        output1, output2 = model(torch.tensor(img1.float()), torch.tensor(img2.float()))
        distance = eucledian_distance(output1,output2)
        for i in range(len(label)):
            logging.info(f"Predicted Eucledian Distance:- {distance[i]}")
            if distance[i] < threshold_euclidean_distance:
                predicted = 1
            else:
                predicted = 0
            logging.info(f"Predicted Label:- {predicted}")
            logging.info(f"Actual Label:- {label[i]}")
            if predicted == label[i]:
                correct += 1
        count = count + 1
        if count == subset_of_batches:
            break
    total_data = subset_of_batches * 64 
    accuracy = correct / total_data
    logging.info(f"Final Accuracy:- {accuracy}")