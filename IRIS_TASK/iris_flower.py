import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# Getting the data
names = ["s_length", "s_width", "p_length", "p_width"]
dataset_class1 = pd.read_csv("IRIS_TASK/class_1", names=names)
dataset_class2 = pd.read_csv("IRIS_TASK/class_2", names=names)
dataset_class3 = pd.read_csv("IRIS_TASK/class_3", names=names)

# making the dataset arrays so they can be used later
dataset1_array = dataset_class1.to_numpy()
dataset2_array = dataset_class2.to_numpy()
dataset3_array = dataset_class3.to_numpy()

# Extracting the first 30 samples for training and the last 20 for testing for Task 1
dataset1_train30 = dataset1_array[0:30]
dataset2_train30 = dataset2_array[0:30]
dataset3_train30 = dataset3_array[0:30]

dataset1_test20 = dataset1_array[30:]
dataset2_test20 = dataset2_array[30:]
dataset3_test20 = dataset3_array[30:]

# Extracting the last 30 samples for training and the first 20 for testing for Task 1d
dataset1_train30_1d = dataset1_array[20:]
dataset2_train30_1d = dataset2_array[20:]
dataset3_train30_1d = dataset3_array[20:]

dataset1_test20_1d = dataset1_array[0:20]
dataset2_test20_1d = dataset2_array[0:20]
dataset3_test20_1d = dataset3_array[0:20]

# Dataset for Task 2 - removing 1 feature
data1_minus_f3 = np.delete(dataset1_array, 2, axis=1)
data2_minus_f3 = np.delete(dataset2_array, 2, axis=1)
data3_minus_f3 = np.delete(dataset3_array, 2, axis=1)
dataset1f3_train30 = data1_minus_f3[0:30]
dataset2f3_train30 = data2_minus_f3[0:30]
dataset3f3_train30 = data3_minus_f3[0:30]
dataset1f3_test20 = data1_minus_f3[30:]
dataset2f3_test20 = data2_minus_f3[30:]
dataset3f3_test20 = data3_minus_f3[30:]
dataset_class1_mf3 = dataset_class1.drop(columns="p_length")
dataset_class2_mf3 = dataset_class2.drop(columns="p_length")
dataset_class3_mf3 = dataset_class3.drop(columns="p_length")

# Dataset for Task 2 - removing 2 features
data1_minus_f2f3 = np.delete(data1_minus_f3, 1, axis=1)
data2_minus_f2f3 = np.delete(data2_minus_f3, 1, axis=1)
data3_minus_f2f3 = np.delete(data3_minus_f3, 1, axis=1)
dataset1f2f3_train30 = data1_minus_f2f3[0:30]
dataset2f2f3_train30 = data2_minus_f2f3[0:30]
dataset3f2f3_train30 = data3_minus_f2f3[0:30]
dataset1f2f3_test20 = data1_minus_f2f3[30:]
dataset2f2f3_test20 = data2_minus_f2f3[30:]
dataset3f2f3_test20 = data3_minus_f2f3[30:]
dataset_class1_mf2f3 = dataset_class1_mf3.drop(columns="s_width")
dataset_class2_mf2f3 = dataset_class2_mf3.drop(columns="s_width")
dataset_class3_mf2f3 = dataset_class3_mf3.drop(columns="s_width")

# Dataset for Task 2 - removing 3 features
data1_minus_f2f3f4 = np.delete(data1_minus_f2f3, 1, axis=1)
data2_minus_f2f3f4 = np.delete(data2_minus_f2f3, 1, axis=1)
data3_minus_f2f3f4 = np.delete(data3_minus_f2f3, 1, axis=1)
dataset1f2f3f4_train30 = data1_minus_f2f3f4[0:30]
dataset2f2f3f4_train30 = data2_minus_f2f3f4[0:30]
dataset3f2f3f4_train30 = data3_minus_f2f3f4[0:30]
dataset1f2f3f4_test20 = data1_minus_f2f3f4[30:]
dataset2f2f3f4_test20 = data2_minus_f2f3f4[30:]
dataset3f2f3f4_test20 = data3_minus_f2f3f4[30:]
dataset_class1_mf2f3f4 = dataset_class1_mf2f3.drop(columns="s_length")
dataset_class2_mf2f3f4 = dataset_class2_mf2f3.drop(columns="s_length")
dataset_class3_mf2f3f4 = dataset_class3_mf2f3.drop(columns="s_length")


def sigmoid(s):
    return 1/(1+np.exp(-s))


def MSE(x, y):
    return ((x-y)**2).mean(axis=1)


def grad_MSE(x, g, t):
    mse_grad = g-t
    g_grad = g*(1-g)
    zk_grad = x.T
    return np.dot(zk_grad, mse_grad*g_grad)


def training(x, t, alpha, iterations):
    W = np.zeros((3, x.shape[1]))
    MSE_values = []
    for i in range(iterations):
        g = sigmoid(np.dot(x, W.T))
        W = W - alpha * grad_MSE(x, g, t).T
        MSE_values.append(MSE(g, t).mean())
    # print(MSE_values)
    return W


def predict(x, W):
    n = x.shape[0]
    prediction = np.array(
        [1.0/(1.0+np.exp(-(np.matmul(W, x[i])))) for i in range(n)])
    return prediction


def generate_confusion_matrix(x, y, W):
    pred = predict(x, W)
    confusion_matrix = np.zeros((3, 3))

    for i in range(pred.shape[0]):
        confusion_matrix[np.argmax(y[i])][np.argmax(pred[i])] += 1
    return confusion_matrix

#print(generate_confusion_matrix(training_dataset_array, train_correct, W))


def plotConfusionMatrix(confusion_matrix_train, confusion_matrix_test, string):
    plt.figure(figsize=(7, 7))
    plt.title("Confusion matrix {}: Training set".format(string))
    sn.heatmap(confusion_matrix_train, annot=True)

    plt.figure(figsize=(7, 7))
    plt.title("Confusion matrix {}: Test set".format(string))
    sn.heatmap(confusion_matrix_test, annot=True)


def trainAndPlotConfusionMatrix(alpha, iterations, string, dataset1_train30, dataset2_train30, dataset3_train30, dataset1_test20, dataset2_test20, dataset3_test20):
    # Combining the training and test datasets to one array for training and one array for testing in task 1d
    training_dataset_array = np.concatenate(
        (dataset1_train30, dataset2_train30, dataset3_train30))
    test_dataset_array = np.concatenate(
        (dataset1_test20, dataset2_test20, dataset3_test20))

    # Making an array with true values of 0,1,2 representing different flowers
    training_true_values = np.array(
        len(dataset1_train30)*[0]+len(dataset2_train30)*[1] + len(dataset3_train30)*[2])

    # Making an array consisting of arrays showing the correct values. This makes the data easier to work with.
    train_correct = np.zeros((training_true_values.shape[0], 3))
    for i, label in np.ndenumerate(training_true_values):
        train_correct[i][label] = 1

    # We are doing the exact same thing to the test array
    test_true_values = np.array(
        len(dataset1_test20)*[0]+len(dataset2_test20)*[1] + len(dataset3_test20)*[2])

    test_correct = np.zeros((test_true_values.shape[0], 3))
    for i, label in np.ndenumerate(test_true_values):
        test_correct[i][label] = 1
    W = training(training_dataset_array, train_correct, 0.007, 50000)
    confusion_matrix_train = generate_confusion_matrix(
        training_dataset_array, train_correct, W)
    confusion_matrix_test = generate_confusion_matrix(
        test_dataset_array, test_correct, W)
    plotConfusionMatrix(confusion_matrix_train, confusion_matrix_test, string)

# Function to show Histograms, only transparent is plotted. Uncomment to see nontransparent ones.


def showHistograms(dataset1, dataset2, dataset3, removed_feature):
    # dataset1.plot.hist(bins=50)
    dataset1.plot.hist(bins=50, alpha=0.7)
    plt.title("Class 1 - Iris Setosa - {}".format(removed_feature))
    plt.xlabel("cm")
    # dataset2.plot.hist(bins=50)
    dataset2.plot.hist(bins=50, alpha=0.7)
    plt.title("Class 2 - Iris Versicolor -{}".format(removed_feature))
    plt.xlabel("cm")
    # dataset3.plot.hist(bins=50)
    dataset3.plot.hist(bins=50, alpha=0.7)
    plt.title("Class 3 - Iris Virginica - {}".format(removed_feature))
    plt.xlabel("cm")


def main():
    alpha = 0.007
    iterations = 50000
    # Task 1 - first 30 for training and last 20 for testing
    string = "Task 1a"
    trainAndPlotConfusionMatrix(alpha, iterations, string, dataset1_train30, dataset2_train30,
                                dataset3_train30, dataset1_test20, dataset2_test20, dataset3_test20)
    # Task1 - first 20 for training and last 20 for testing
    string = "Task 1d"
    trainAndPlotConfusionMatrix(alpha, iterations, string, dataset1_train30_1d, dataset2_train30_1d,
                                dataset3_train30_1d, dataset1_test20_1d, dataset2_test20_1d, dataset3_test20_1d)

    # TASK 2
    # plot histograms
    removed_feature = "All features displayed"
    showHistograms(dataset_class1, dataset_class2,
                   dataset_class3, removed_feature)

    # Remove 1 feature
    string = "Task 2 - minus feature 3"
    trainAndPlotConfusionMatrix(alpha, iterations, string, dataset1f3_train30, dataset2f3_train30,
                                dataset3f3_train30, dataset1f3_test20, dataset2f3_test20, dataset3f3_test20)
    removed_feature = "feature 3 removed"
    showHistograms(dataset_class1_mf3, dataset_class2_mf3,
                   dataset_class3_mf3, removed_feature)

    # Remove 2 features
    string = "Task 2 - minus feature 2 and 3"
    trainAndPlotConfusionMatrix(alpha, iterations, string, dataset1f2f3_train30, dataset2f2f3_train30,
                                dataset3f2f3_train30, dataset1f2f3_test20, dataset2f2f3_test20, dataset3f2f3_test20)
    removed_feature = "feature 3 and 2 removed"
    showHistograms(dataset_class1_mf2f3,
                   dataset_class2_mf2f3, dataset_class3_mf2f3, removed_feature)

    # Remove 3 features
    string = "Task 2 - minus feature 2, 3 and 4"
    trainAndPlotConfusionMatrix(alpha, iterations, string, dataset1f2f3f4_train30, dataset2f2f3f4_train30,
                                dataset3f2f3f4_train30, dataset1f2f3f4_test20, dataset2f2f3f4_test20, dataset3f2f3f4_test20)
    # Plot of the histograms
    removed_feature = "feature 3,2 and 1 removed"
    showHistograms(dataset_class1_mf2f3f4,
                   dataset_class2_mf2f3f4, dataset_class3_mf2f3f4, removed_feature)
    plt.show()


if __name__ == "__main__":
    main()
