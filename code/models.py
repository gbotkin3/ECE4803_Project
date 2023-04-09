# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score

def KNN(traindata, trainlabels, testdata):

    neigh = KNeighborsClassifier(n_neighbors=50).fit(traindata, trainlabels.reshape(-1))

    return neigh.predict(testdata)

def Decision_Tree(traindata, trainlabels, testdata):

    clf = DecisionTreeClassifier(min_weight_fraction_leaf = 0.1, random_state=0).fit(traindata, trainlabels.reshape(-1))

    return clf.predict(testdata)

def gNB(traindata, trainlabels, testdata):

    gm = GaussianNB().fit(traindata, trainlabels.reshape(-1))

    return gm.predict(testdata)

def CNN(trainloader, validloader, testloader, epochs = 100, lr = 1e-3, train = False): 

    class OCTCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(

                nn.Conv2d(1, 8, kernel_size=3, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(4,4),
                nn.Conv2d(8, 16, kernel_size=3, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(16*11*60, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 3),
                nn.ReLU())
        
        def forward(self, x):

            # for layer in self.network:
            #     x = layer(x)
            #     print(x.size())

            # return x

            return self.network(x)

    net = OCTCNN().to("cuda") ## Init the network

    loss_function = torch.hub.load(
	'adeelh/pytorch-multi-class-focal-loss',
    alpha = [0.5, 0, 1],
	model='focal_loss',
	gamma=2,
	reduction='mean',
	device='cuda',
	dtype=torch.float32,
	force_reload=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = 0.001)  ## Using AdamW Optimizer
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(trainloader), epochs=epochs)

    if train == True:

        print("Training CNN")

        best_accuracy = 0.0

        for epoch in range(epochs):

            net.train()  # training mode

            total_loss = 0

            for iteration, (x, y) in enumerate(trainloader):

                x, y = x.to("cuda", non_blocking=True), y.to("cuda", non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                out = net(x)
                loss = loss_function(out, y)

                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())

                #print("Epoch: ", epoch, " Iteration: ", iteration)

            labels_train = []
            predictions_train = []

            labels_valid = []
            predictions_valid = []

            labels_test = []
            predictions_test = []

            with torch.no_grad():
            
                net.eval() 
                for (x, y) in trainloader:

                    x, y = x.to("cuda"), y.to("cuda")

                    output = net(x)

                    _, predicted = torch.max(output.data, 1)

                    labels_train += y.cpu().numpy().tolist()

                    predictions_train += predicted.cpu().numpy().tolist()

                    train_accuracy = balanced_accuracy_score(labels_train, predictions_train)

                for (x, y) in validloader:

                    x, y = x.to("cuda"), y.to("cuda")

                    output = net(x)

                    _, predicted = torch.max(output.data, 1)

                    labels_valid += y.cpu().numpy().tolist()

                    predictions_valid += predicted.cpu().numpy().tolist()

                    validation_accuracy = balanced_accuracy_score(labels_valid, predictions_valid)

                for (x, y) in testloader:

                    x, y = x.to("cuda"), y.to("cuda")

                    output = net(x)

                    _, predicted = torch.max(output.data, 1)

                    labels_test += y.cpu().numpy().tolist()

                    predictions_test += predicted.cpu().numpy().tolist()

                    test_accuracy = balanced_accuracy_score(labels_test, predictions_test)


            #scheduler.step()

            print('Epoch : {} | Total Training Loss : {:0.4f} | Test Accuracy: {:0.4f} | Train Accuracy: {:0.4f} | Validation Accuracy: {:0.4f}'.format(epoch, total_loss, test_accuracy, train_accuracy, validation_accuracy))
            
            if test_accuracy >= best_accuracy:
                torch.save(net.state_dict(), "../results/model_state_dict")
                best_accuracy = validation_accuracy

            #if validation_accuracy > 0.99 or (train_accuracy - validation_accuracy) > 0.05:
                #break

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

            #print(output)

            _, predicted = torch.max(output.data, 1)

            labels += y.cpu().numpy().tolist()

            predictions += predicted.cpu().numpy().tolist()

        #print("Predictions: ", len(predictions), "Actual: ", len(labels))

    return predictions, labels
