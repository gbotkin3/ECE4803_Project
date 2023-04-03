# Methods for Reporting Model Performance

from sklearn.metrics import balanced_accuracy_score

def check_performance(model_name, predictions, labels):

    print(model_name + 's Performance')

    accuracy_results = balanced_accuracy_score(labels, predictions)

    print("Accuracy: ", accuracy_results, "\n")

    results = open("../results/Models_Performance", "a")

    results.write(model_name + "\n")
    results.write("Accuracy: " + str(accuracy_results) + "\n")

    results.close()

    return

def AUROC(predictions, labels): ## TODO

    return

def precision(predictions, labels):  ## TODO

    return

def recall(predictions, labels):  ## TODO

    return

def f1(predictions, labels):  ## TODO

    return