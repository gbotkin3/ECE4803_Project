# Methods for Reporting Model Performance

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

def check_performance(model_name, predictions, labels):

    print(model_name + 's Performance')

    accuracy_results = accuracy_score(labels, predictions)
    balanced_accuracy_results = balanced_accuracy_score(labels, predictions)

    precision = precision_score(labels, predictions, average = 'weighted')
    recall = recall_score(labels, predictions, average = 'weighted')
    F1 = f1_score(labels, predictions,average = 'weighted')

    print("Accuracy: ", accuracy_results, "")
    print("Balanced Accuracy Results: ", balanced_accuracy_results, "")
    print("Precision: ", precision, "")
    print("Recall: ", recall, "")
    print("F1: ", F1, "\n")

    results = open("../results/Models_Performance", "a")

    results.write(model_name + "\n")
    results.write("Accuracy: " + str(accuracy_results) + "\n")
    results.write("Balanced Accuracy Results: " + str(balanced_accuracy_results) + "\n")
    results.write("Precision: " + str(precision) + "\n")
    results.write("Recall: " + str(recall) + "\n")
    results.write("F1: " + str(F1) + "\n")

    results.close()

    return