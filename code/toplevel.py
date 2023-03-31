# Import Python Files Containing Methods

import visualization as vis ## Contains Methods for Visulizing Data
import models as mod        ## Contains Methods for Training and Testing Models on Provided Train and Test Datasets
import performance as per   ## Contains Methods for Reporting Model Performance

import random
from time import sleep
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import pandas as pd

from PIL import Image

import argparse
import os
import copy

# Important Constants used to control Settings

## Batch Sizes (use -1 to use all samples)
train_batch_size = 1000
test_batch_size = 1000

## Visualization

visualize = False
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
    trainset_tensor = OCTDataset_Tensor(args, 'train', transform=None)
    testset_tensor = OCTDataset_Tensor(args, 'test', transform=None)

if train_batch_size == -1:
    train_batch_size = len(trainset)

if test_batch_size == -1:
    test_batch_size = len(testset)

trainloader = DataLoader(trainset, batch_size=train_batch_size)
testloader = DataLoader(testset, batch_size=test_batch_size)

## Create a random dataset of just the train data values without labels

print("\n Sampling Trainset:")

trainset_data = next(iter(trainloader))[0].numpy()
trainset_data = trainset_data.reshape(trainset_data.shape[0], trainset_data.shape[2])

trainset_labels = next(iter(trainloader))[1].numpy()
trainset_labels = trainset_labels.reshape((-1, 1))

print("Trainset Sampled \n")

## Normalize and Standardize Training Dataset

scaler = StandardScaler()
trainset_data = scaler.fit_transform(trainset_data) 

## Run Training through the PCA algorithm

pca = PCA(n_components=3)
trainset_data = pca.fit_transform(trainset_data) 

## Create a random dataset of just the test data values without labels

print("Sampling Testset:")

testset_data = next(iter(testloader))[0].numpy()
testset_data = testset_data.reshape(testset_data.shape[0], testset_data.shape[2])

testset_labels = next(iter(testloader))[1].numpy()
testset_labels = testset_labels.reshape((-1, 1))

print("Testset Sampled \n")

## Normalize and Standardize Training Dataset
testset_data = scaler.transform(testset_data) 

## Run Training through the PCA algorithm
testset_data = pca.transform(testset_data) 

# Visualize  the Data

if visualize:
    print("Visualizing Data")
    
    if show_scatterplot:
        vis.scatter_plot_all(trainset_data, trainset_labels, show_plot)
    if show_kde:
        vis.kde_map_all(trainset_data, trainset_labels, show_plot)
    if show_pairplot:
        vis.pairplot_all(trainset_data, trainset_labels, show_plot)

    print("Data Visualized \n")
else:
    print("Data Visualization Disabled \n")

# Models

print("Running Models")

if run_knn:
    predictions = mod.KNN(trainset_data, trainset_labels, testset_data)
    per.check_performance("KNN", predictions, testset_labels)
else:
    print("KNN Disabled")

if run_decision_tree:
    predictions = mod.Decision_Tree(trainset_data, trainset_labels, testset_data)
    per.check_performance("Decision Tree", predictions, testset_labels)
else:
    print("Decision Tree Disabled")

if run_gmm:
    predictions = mod.GMM(trainset_data, trainset_labels, testset_data)
    per.check_performance("GMM", predictions, testset_labels)
else:
    print("GMM Disabled")

if run_cnn:
    predictions = mod.CNN()
    per.check_performance("CNN", predictions, testset_labels)
else:
    print("CNN Disabled")

print("Models Ran \n")
