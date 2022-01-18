import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

names = ["s_length", "s_width", "p_length", "p_width"]
dataset_class1 = pd.read_csv("class_1", names=names)
dataset_class2 = pd.read_csv("class_2", names=names)
dataset_class3 = pd.read_csv("class_3", names=names)

#print("Iris dataset class_1: \n", dataset_class1)
#print("Iris dataset class_2: \n", dataset_class2)
#print("Iris dataset class_3: \n", dataset_class3)

#making the dataset arrays so they can be used later
dataset1_array = dataset_class1.to_numpy()
dataset2_array = dataset_class2.to_numpy()
dataset3_array = dataset_class3.to_numpy()
#print(dataset1_array)
#Now the dataset is an array consisting of multiple arrays for each samples

dataset1_train30 = dataset1_array[0:30]
dataset2_train30 = dataset2_array[0:30]
dataset3_train30 = dataset3_array[0:30]

dataset1_test20 = dataset1_array[30:]
dataset2_test20 = dataset2_array[30:]
dataset3_test20 = dataset3_array[30:]

#training and test datasets
training_dataset_array = np.concatenate((dataset1_train30, dataset2_train30, dataset3_train30))
test_dataset_array = np.concatenate((dataset1_test20, dataset2_test20, dataset3_test20))

#array with true values of 0,1,2 representing different flowers
training_true_values = np.array(len(dataset1_train30)*[0]+len(dataset2_train30)*[1]+ len(dataset3_train30)*[2])

train_correct = np.zeros((training_true_values.shape[0],3))
for i, label in np.ndenumerate(training_true_values):
    train_correct[i][label] = 1

def sigmoid(s):
    return 1/(1+np.exp(-s))

def MSE(x,y):
    return ((x-y)**2).mean(axis=1)

def grad_MSE(x,g,t):
    mse_grad = g-t
    g_grad = g*(1-g)
    zk_grad = x.T
    return np.dot(zk_grad, mse_grad*g_grad)

def training(x,t,alpha,iterations):
    W = np.zeros((3,x.shape[1]))
    MSE_values = []
    for i in range(iterations):
        g = sigmoid(np.dot(x,W.T))
        W = W - alpha * grad_MSE(x,g,t).T
        MSE_values.append(MSE(g,t).mean())
    return W

W = training(training_dataset_array, train_correct, 0.007, 500)

def predict(x, W):
    n = x.shape[0]
    prediction = np.array(
        [1.0/(1.0+np.exp(-(np.matmul(W,x[i])))) for i in range(n)])
    return prediction


predicted_dataset = predict(training_dataset_array, W)

def generate_confusion_matrix(x, y, W):
    pred = predict(x, W)
    confusion_matrix = np.zeros((3, 3))

    for i in range(pred.shape[0]):
        confusion_matrix[np.argmax(y[i])][np.argmax(pred[i])] += 1
    return confusion_matrix

print(generate_confusion_matrix(training_dataset_array, train_correct, W))
plt.figure(figsize = (7,7))
plt.title("Confusion matrix:")
sn.heatmap(generate_confusion_matrix(training_dataset_array, train_correct, W), annot=True)
plt.show()