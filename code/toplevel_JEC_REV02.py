# Import Python Files Containing Methods
###This is the very first program version that has a running CNN model###

import visualization as vis ## Contains Methods for Visulizing Data
import models as mod        ## Contains Methods for Training and Testing Models on Provided Train and Test Datasets
import performance as per   ## Contains Methods for Reporting Model Performance

import random
from time import sleep
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import pandas as pd

from PIL import Image

import argparse
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# Important macros to control settings

## Batch Sizes (use -1 to use all samples)
train_batch_size = 3000
test_batch_size = 1000

## Visualization

visualize = True
show_plot = True

show_scatterplot = True
show_kde = True
show_pairplot = True

## Models

run_knn = True
run_decision_tree = True
run_gmm = True
run_cnn = False

# Create train and test datasets using provide dataloader

LABELS_Severity = {35: 0,
                   43: 0,
                   47: 1,
                   53: 1,
                   61: 2,
                   65: 2,
                   71: 2,
                   85: 2}

mean = (.1706)
std = (.2112)
normalize = transforms.Normalize(mean=mean, std=std)

transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    normalize,
])

#Transform for resizing images. It is used by the CNN
hor_img_res = 202
vert_img_res = 202
resized_img_len=hor_img_res*vert_img_res
transform_img = transforms.Compose([
    transforms.Resize(size=(hor_img_res,vert_img_res))
])

class OCTDataset_Numpy(Dataset):
    def __init__(self, args, subset='train', transform=None,):
        if subset == 'train':
            self.annot = pd.read_csv(args.annot_train_prime)
        elif subset == 'test':
            self.annot = pd.read_csv(args.annot_test_prime)
            
        self.annot['Severity_Label'] = [LABELS_Severity[drss] for drss in copy.deepcopy(self.annot['DRSS'].values)] 
        # print(self.annot)
        self.root = os.path.expanduser(args.data_root)
        self.transform = transform
        # self.subset = subset
        self.nb_classes=len(np.unique(list(LABELS_Severity.values())))
        self.path_list = self.annot['File_Path'].values
        self._labels = self.annot['Severity_Label'].values
        assert len(self.path_list) == len(self._labels)
        # idx_each_class = [[] for i in range(self.nb_classes)]

    def __getitem__(self, index):

        # Retrieve Img and Label from CSV and Data Files
        img, target = Image.open(self.root+self.path_list[index]).convert("L"), self._labels[index]
        img = np.array(img)

        # Rescale img pixels to 0 - 1
        img = img / 255
        # Reshape image in to 1xP (P = # of Pixels)
        img = img.reshape(1, -1)

        return img, target

    def __len__(self):
        return len(self._labels)         

class OCTDataset_Tensor(Dataset):
    def __init__(self, args, subset='train', transform=None,):
        if subset == 'train':
            self.annot = pd.read_csv(args.annot_train_prime)
        elif subset == 'test':
            self.annot = pd.read_csv(args.annot_test_prime)
            
        self.annot['Severity_Label'] = [LABELS_Severity[drss] for drss in copy.deepcopy(self.annot['DRSS'].values)] 
        # print(self.annot)
        self.root = os.path.expanduser(args.data_root)
        self.transform = transform
        # self.subset = subset
        self.nb_classes=len(np.unique(list(LABELS_Severity.values())))
        self.path_list = self.annot['File_Path'].values
        self._labels = self.annot['Severity_Label'].values
        assert len(self.path_list) == len(self._labels)
        # idx_each_class = [[] for i in range(self.nb_classes)]

    def __getitem__(self, index):
        img, target = Image.open(self.root+self.path_list[index]).convert("L"), self._labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self._labels)    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_train_prime', type = str, default = '../data_files/df_prime_train.csv')
    parser.add_argument('--annot_test_prime', type = str, default = '../data_files/df_prime_test.csv')
    parser.add_argument('--data_root', type = str, default = '../data_files/')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    trainset = OCTDataset_Numpy(args, 'train', transform=None)
    testset = OCTDataset_Numpy(args, 'test', transform=None)
    trainset_tensor = OCTDataset_Tensor(args, 'train', transform=transform)
    testset_tensor = OCTDataset_Tensor(args, 'test', transform=transform)

#if train_batch_size == -1:
#    train_batch_size = len(trainset)

#if test_batch_size == -1:
#    test_batch_size = len(testset)

#trainloader = DataLoader(trainset, batch_size=train_batch_size)
#testloader = DataLoader(testset, batch_size=test_batch_size)

## Create a random dataset of just the train data values without labels

print("\n Sampling Trainset:")

#trainset_data = next(iter(trainloader))[0].numpy()
#trainset_data = trainset_data.reshape(trainset_data.shape[0], trainset_data.shape[2])

#trainset_labels = next(iter(trainloader))[1].numpy()
#trainset_labels = trainset_labels.reshape((-1, 1))

print("Trainset Sampled \n")

## Normalize and Standardize Training Dataset

#scaler = StandardScaler()
#trainset_data = scaler.fit_transform(trainset_data)

## Run Training through the PCA algorithm

#pca = PCA(n_components=3)
#trainset_data = pca.fit_transform(trainset_data)

## Create a random dataset of just the test data values without labels

print("Sampling Testset:")

#testset_data = next(iter(testloader))[0].numpy()
#testset_data = testset_data.reshape(testset_data.shape[0], testset_data.shape[2])

#testset_labels = next(iter(testloader))[1].numpy()
#testset_labels = testset_labels.reshape((-1, 1))

print("Testset Sampled \n")

## Normalize and Standardize Test Dataset
#testset_data = scaler.transform(testset_data)

## Run Test Data through the PCA algorithm
#testset_data = pca.transform(testset_data)

##Load Trainset for Tensor
trainset_tensor_data_array = np.zeros((len(trainset_tensor), resized_img_len))
for i in (range(0, 5000)):
    img_current = Image.open(trainset_tensor.root+trainset_tensor.path_list[i]).convert("L")
    img_current = transform_img(img_current)                                                #Resizes image
    img_current_arr = np.ndarray.flatten(np.array(img_current))
    trainset_tensor_data_array[i:i+1] = img_current_arr

X = trainset_tensor_data_array
y = trainset_tensor._labels
print("Full Trainset data shape:", X.shape, "Trainset data type:",type(X))
print("Full Trainset labels shape:", y.shape, "Trainset labels type:",type(y))

# create train test splits
num_train_cnn = 500
#num_test = X.shape[0] - num_train
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=num_test, train_size=num_train, random_state=4803)
X_train = X[0:num_train_cnn,:]
y_train =y[0:num_train_cnn]

print("Trainset data shape 'X_train':", X_train.shape, "Trainset data type 'X_train':",type(X_train))
print("Trainset labels shape 'y_train':", y_train.shape, "Trainset labels type 'y_train':",type(y_train))

class OCTData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __getitem__(self, index):
        input = self.X[index].reshape(202, 202)
        input = np.expand_dims(input, axis=0)
        input = torch.tensor(input, dtype=torch.float)

        target = self.y[index]
        target = torch.tensor(target, dtype=torch.long)

        return input, target

    def __len__(self):
        return self.X.shape[0]

train_batch_size_cnn = 100
train_dataset_cnn = OCTData(X_train, y_train)
trainloader_cnn = DataLoader(train_dataset_cnn, batch_size=train_batch_size_cnn, shuffle=True)

print("\nTRAIN DATASET")
print("Train Dataset Type (data):", type(train_dataset_cnn.X))
print("Train Dataset Shape (data):", train_dataset_cnn.X.shape)
#print("Get Item Method Test (data): \n", train_dataset.X[0], "\n")

print("Train Dataset Type (labels):", type(train_dataset_cnn.y))
print("Train Dataset Shape (labels):", train_dataset_cnn.y.shape)
#print("Get Item Method Test (labels):", train_dataset.y[0], "\n")

print("\nTRAINLOADER")
print("Train Dataset Loader Type:", type(trainloader_cnn))
print("Train Dataset trainloader length:", len(trainloader_cnn))
#print("Get Item Method Test (labels):", train_dataset.y[0], "\n")


class OCTCNN(torch.nn.Module):
    def __init__(self):
        super(OCTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=0)
        #self.pool = nn.MaxPool2d(4,4)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        batch_size = x.shape[0]
        #print("\nBatch Size:", batch_size)
        #print("x.shape:", x.shape)
        a1 = F.relu(self.conv1(x))
        #print("a1.shape:", a1.shape)
        a2 = F.max_pool2d(a1, (4, 4))
        #print("a2.shape:", a2.shape)
        b = F.max_pool2d(F.relu(self.conv2(a2)),(10,10))
        #print("b.shape:", b.shape)
        c = b.view(-1, 16*5*5)
        #c = x.view(-1, self.num_flat_features(b))
        d = F.relu(self.fc1(c))
        e = F.relu(self.fc2(d))
        f = self.fc3(e)
        #g = self.linear(f.reshape(batch_size, -1))
        return f

    def num_flat_features(selfself,x):
        size=x.size()[1:]
        num_features = 1
        for s in size:
            num_features*=s
        return num_features


lr = 1e-3
epochs = 100
net = OCTCNN() # initliaze the network

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)  ## Using Adam Optimizer

for epoch in range(epochs):

    net.train()  # training mode

    for iteration, (x, y) in enumerate(trainloader_cnn):
        optimizer.zero_grad()
        out = net(x)
        loss = loss_function(out, y)

        loss.backward()
        optimizer.step()

        print('Epoch : {} | Training Loss : {:0.4f}'.format(epoch, loss.item()))


#Testing of Array contents (index 3512 is only an example. It could be any index within the range of the dataset 24252
img_3512 = Image.open(trainset_tensor.root+trainset_tensor.path_list[3512]).convert("L")
img_3512 = transform_img(img_3512)
img_current_arr_3512 = np.ndarray.flatten(np.asarray(img_3512))
print("img_3512 shape:", img_3512.size)
img_3512.show()

img_3512_recons=trainset_tensor_data_array[3512,:]
print("Image 3512 Shape:", img_3512_recons.shape, type(img_3512_recons))
img_3512_recons_res = img_3512_recons.reshape((hor_img_res,vert_img_res))
print("img_3512_recons_res shape:", img_3512_recons_res.shape)
img_3512_recons_PIL=Image.fromarray(img_3512_recons_res)
img_3512_recons_PIL.show()

print("Index of Max at img_current_arr_3512:", np.argmax(img_current_arr_3512))
print("Value:", img_current_arr_3512[np.argmax(img_current_arr_3512)])
print("Index of Max at train_data_arr[3512]:", np.argmax(img_3512_recons))
print("Value:", img_3512_recons[np.argmax(img_3512_recons)])


