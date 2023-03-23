# Methods for Reporting Model Performance

def check_performance(model_name, predictions, labels):

    print(model_name + 's Performance')

    accuracy_results = accuracy(predictions, labels)

    print("Accuracy: ", accuracy_results, "\n")

    results = open("../results/Models_Performance", "a")

    results.write(model_name + "\n")
    results.write("Accuracy: " + str(accuracy_results) + "\n")

    results.close()

    return


def accuracy(predictions, labels):

    truths = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            truths += 1
    accuracy = truths / len(predictions)

    return accuracy

def AUROC(predictions, labels): ## TODO

    return

def precision(predictions, labels):  ## TODO

    return

def recall(predictions, labels):  ## TODO

    return

def f1(predictions, labels):  ## TODO

    return