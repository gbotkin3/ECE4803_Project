# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import torch
import torch.nn as nn
import torch.nn.functional as F

def KNN(traindata, trainlabels, testdata):

    neigh = KNeighborsClassifier(n_neighbors=50).fit(traindata, trainlabels.reshape(-1))

    return neigh.predict(testdata)

def Decision_Tree(traindata, trainlabels, testdata):

    clf = DecisionTreeClassifier(min_weight_fraction_leaf = 0.1, random_state=0).fit(traindata, trainlabels.reshape(-1))

    return clf.predict(testdata)

def gNB(traindata, trainlabels, testdata):

    gm = GaussianNB().fit(traindata, trainlabels.reshape(-1))

    return gm.predict(testdata)

def CNN(trainloader, testloader, epochs = 100, lr = 1e-3, train = False): 

    class OCTCNN(torch.nn.Module):
        def __init__(self):
            super(OCTCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=0)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(16*5*5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 3)

        def forward(self, x):

            #print(x.shape)
            layer_in_1 = F.relu(self.conv1(x))
            #print(layer_in_1.shape)
            layer_1_2 = F.max_pool2d(layer_in_1, (4, 4))
            #print(layer_1_2.shape)
            layer_2_3 = F.max_pool2d(F.relu(self.conv2(layer_1_2)), (10,10))
            #print(layer_2_3.shape)
            layer_3_4 = layer_2_3.view(-1, 16*5*5)
            #print(layer_3_4.shape)
            layer_4_6 = F.relu(self.fc1(layer_3_4))
            #print(layer_4_6.shape)
            layer_6_7 = F.relu(self.fc2(layer_4_6))
            #print(layer_6_7.shape)
            layer_7_out = self.fc3(layer_6_7)
            #print(layer_7_out.shape)
            return layer_7_out

    net = OCTCNN().to("cuda") ## Init the network

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  ## Using Adam Optimizer

    if train == True:

        print("Training CNN")

        net.train()  # training mode

        for epoch in range(epochs):

            total_loss = 0

            for iteration, (x, y) in enumerate(trainloader):

                x, y = x.to("cuda"), y.to("cuda")

                optimizer.zero_grad(set_to_none=True)

                out = net(x)
                loss = loss_function(out, y)

                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())

                #print("Epoch: ", epoch, " Iteration: ", iteration)

            print('Epoch : {} | Total Training Loss : {:0.4f}'.format(epoch, total_loss))

        torch.save(net.state_dict(), "../results/model_state_dict")

    else:

        print("Loading CNN Weights")
        net.load_state_dict(torch.load("../results/model_state_dict"))

    print("Testing CNN")

    labels = []
    predictions = []

    with torch.no_grad():
        net.eval() 
        for (x, y) in testloader:

            x, y = x.to("cuda"), y.to("cuda")

            output = net(x)
            _, predicted = torch.max(output.data, 1)

            labels += y.cpu().numpy().tolist()

            predictions += predicted.cpu().numpy().tolist()

        
        #print("Predictions: ", len(predictions), "Actual: ", len(labels))

    return predictions, labels
