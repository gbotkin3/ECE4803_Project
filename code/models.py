# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB

def KNN(traindata, trainlabels, testdata):

    neigh = KNeighborsClassifier(n_neighbors=3).fit(traindata, trainlabels.reshape(-1))

    return neigh.predict(testdata)

def Decision_Tree(traindata, trainlabels, testdata):

    clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0).fit(traindata, trainlabels.reshape(-1))

    return clf.predict(testdata)

def GMM(traindata, trainlabels, testdata):

    #gm = GaussianMixture(n_components=3, random_state=1337).fit(traindata)
    gm = GaussianNB().fit(traindata, trainlabels.reshape(-1))

    return gm.predict(testdata)

def CNN():  ## TODO

    predictions = 0

    return predictions
