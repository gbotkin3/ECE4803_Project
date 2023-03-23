# Import Python Files Containing Methods

import visualization as vis ## Contains Methods for Visulizing Data
import models as mod        ## Contains Methods for Training and Testing Models on Provided Train and Test Datasets
import performance as per   ## Contains Methods for Reporting Model Performance

import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import SequentialSampler
from torchvision import transforms

import numpy as np
import pandas as pd

from PIL import Image

from sklearn.decomposition import PCA

import argparse
import os
import copy

# Important Constants used to control Settings

## Batch Sizes (use -1 to use all samples)
train_batch_size = 100
test_batch_size = 100

## Visualization

visualize = True
show_plot = True

## Models

run_knn = True
run_decision_tree = True
run_gmm = True
run_cnn = False

# Create train and test datasets using provide dataloader
# Used PCA to reduce image features from 224*224 (50176) to 3 for visualization and training purposes

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
    
class OCTDataset(Dataset):
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

        # Normalize Image Features

        img = img / 255

        # Run Image through PCA algorithm to reduce features to 3
        pca = PCA(n_components=3)
        pca.fit(img) 
        img = pca.singular_values_.reshape(1, -1)

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
    trainset = OCTDataset(args, 'train', transform=None)
    testset = OCTDataset(args, 'test', transform=None)

# Create a random dataset of just the train data values without labels

print("Sampling Trainset:")

if (train_batch_size == -1):
    train_batch_size = len(trainset)

trainset_data = np.zeros((train_batch_size, len(trainset[0][0][0])))
trainset_labels = np.zeros((train_batch_size, 1))

j = 0
for i in random.sample(range(0, len(trainset)), train_batch_size):
    
    trainset_data[j] = trainset[i][0][0]
    trainset_labels[j] = trainset[i][1]

    j += 1

    if (j % (train_batch_size * 0.05) == 0):
        print(j / train_batch_size * 100, "%")

print("100 %")
print("Trainset Sampled \n")

# Create a random dataset of just the test data values without labels

print("Sampling Testset:")

if (test_batch_size == -1):
    test_batch_size = len(testset)

testset_data = np.zeros((test_batch_size, len(testset[0][0][0])))
testset_labels = np.zeros((test_batch_size, 1))

j = 0
for i in random.sample(range(0, len(testset)), test_batch_size):
    
    testset_data[j] = testset[i][0][0]
    testset_labels[j] = testset[i][1]
    
    j += 1

    if (j % (test_batch_size * 0.05) == 0):
        print(j / test_batch_size * 100, "%")

print("100 %")
print("Testset Sampled \n")

# Visualize  the Data

if visualize:
    print("Visualizing Data")
    
    vis.scatter_plot_all(trainset_data, trainset_labels, show_plot)
    vis.kde_map_all(trainset_data, trainset_labels, show_plot)
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
