# Import Python Files Containing Methods

import visualization as vis ## Contains Methods for Visulizing Data
import models as mod        ## Contains Methods for Training and Testing Models on Provided Train and Test Datasets
import performance as per   ## Contains Methods for Reporting Model Performance

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms

import numpy as np
import pandas as pd

from PIL import Image

import argparse
import os
import copy

# Important macros to control settings

## Batch Sizes (use -1 to use all samples)

### KNN / DT / GNB
train_batch_size = -1
test_batch_size = -1

### CNN

train_batch_size_tensor = 500
test_batch_size_tensor =  100

## Visualization

visualize = True
show_plot = False

show_scatterplot = True
show_kde = True
show_pairplot = True

## Models

run_knn = True
run_decision_tree = True
run_gnb = True

run_cnn = True
train_cnn = False

# Create Train and Test datasets

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
    
transform = transforms.Compose([
    #transforms.Resize(size=(100,100)),
    transforms.CenterCrop(size = (100, 496)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
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

        #print(img.size)

        #img = transforms.Resize(size = (100, 500)).forward(img)
        img = transforms.CenterCrop(size = (100, 496)).forward(img)
        
        img = np.array(img)
        
        # Reshape image in to 1xP (P = # of Pixels)
        img = img.reshape(1, -1)

        # Rescale img pixels to 0 - 1
        img = img / 255
        
        # Normalize Img Pixels
        img = (img - mean) / std

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

print("")

if train_batch_size == -1:
    train_batch_size = len(trainset)

if test_batch_size == -1:
    test_batch_size = len(testset)

if train_batch_size_tensor == -1:
    train_batch_size_tensor = len(trainset_tensor)

if test_batch_size_tensor == -1:
    test_batch_size_tensor = len(testset_tensor)

## Create Dataloaders

print("Creating Dataloaders")

if run_knn or run_decision_tree or run_gnb:
    trainloader = DataLoader(trainset, batch_size=train_batch_size)
    testloader = DataLoader(testset, batch_size=test_batch_size)

if run_cnn:

    train, valid = random_split(trainset_tensor, [int(0.8*len(trainset_tensor)), int(0.2*len(trainset_tensor)) + 1])

    trainloader_tensor = DataLoader(train, batch_size=train_batch_size_tensor, num_workers=4, pin_memory=True, shuffle=True)
    
    validloader_tensor = DataLoader(valid, batch_size=train_batch_size_tensor, num_workers=4, pin_memory=True, shuffle=False)
    testloader_tensor = DataLoader(testset_tensor, batch_size=test_batch_size_tensor, num_workers=4, pin_memory=True, shuffle=False)

print("Dataloaders Created\n")

## Create a random dataset of just the train data values without labels

if run_knn or run_decision_tree or run_gnb:
    print("Sampling Trainset")

    trainset_data = next(iter(trainloader))[0].numpy()
    trainset_data = trainset_data.reshape(trainset_data.shape[0], trainset_data.shape[2])

    trainset_labels = next(iter(trainloader))[1].numpy()
    trainset_labels = trainset_labels.reshape((-1, 1))

    print("Trainset Sampled \n")

    ## Normalize and Standardize Training Dataset

    print("Scaling Trainset")

    scaler = StandardScaler()
    trainset_data = scaler.fit_transform(trainset_data) 

    ## Run Training through the PCA algorithm

    print("Running Trainset through PCA with n=3")

    pca = PCA(n_components=3)
    trainset_data = pca.fit_transform(trainset_data) 

    print("PCA Variance: ", pca.explained_variance_ratio_, "\n")

    ## Create a random dataset of just the test data values without labels

    print("Sampling Testset")

    testset_data = next(iter(testloader))[0].numpy()
    testset_data = testset_data.reshape(testset_data.shape[0], testset_data.shape[2])

    testset_labels = next(iter(testloader))[1].numpy()
    testset_labels = testset_labels.reshape((-1, 1))

    print("Testset Sampled \n")

    ## Normalize and Standardize Test Dataset
    print("Scaling Testset")
    testset_data = scaler.transform(testset_data) 

    ## Run Test Data through the PCA algorithm
    print("Running Testset through PCA with n=3\n")
    testset_data = pca.transform(testset_data) 

# Visualize the Data

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

print("Running Models\n")

if run_knn:
    print("KNN Enabled\n")
    predictions = mod.KNN(trainset_data, trainset_labels, testset_data)
    per.check_performance("KNN", predictions, testset_labels)
else:
    print("KNN Disabled\n")

if run_decision_tree:
    print("Decision Tree Enabled\n")
    predictions = mod.Decision_Tree(trainset_data, trainset_labels, testset_data)
    per.check_performance("Decision Tree", predictions, testset_labels)
else:
    print("Decision Tree Disabled\n")

if run_gnb:
    print("gNB Enabled\n")
    predictions = mod.gNB(trainset_data, trainset_labels, testset_data)
    per.check_performance("gNB", predictions, testset_labels)
else:
    print("GMM Disabled\n")

if run_cnn:
    print("CNN Enabled\n")
    predictions, labels = mod.CNN(trainloader_tensor, validloader_tensor, testloader_tensor, epochs = 1000, lr = 1e-3, train = train_cnn)
    per.check_performance("CNN", predictions, labels)
else:
    print("CNN Disabled\n")

print("Models Ran")
