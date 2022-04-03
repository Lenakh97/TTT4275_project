import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

names = ["s_length", "s_width", "p_length", "p_width"]
dataset_class1 = pd.read_csv("/home/lena/Documents/Estimering/Project/IRIS_TASK/class_1", names=names)
dataset_class2 = pd.read_csv("/home/lena/Documents/Estimering/Project/IRIS_TASK/class_2", names=names)
dataset_class3 = pd.read_csv("/home/lena/Documents/Estimering/Project/IRIS_TASK/class_3", names=names)

#dataset_class1.hist()
#dataset_class2.hist()
#dataset_class3.hist()
plt.figure(figsize = (7,7))
#dataset_class1.plot.hist(bins=50)
#dataset_class1.plot.hist(bins=50, alpha=0.7)
#dataset_class2.plot.hist(bins=50)
#dataset_class2.plot.hist(bins=50, alpha=0.7)
#dataset_class3.plot.hist(bins=50)
#dataset_class3.plot.hist(bins=50, alpha=0.7)
dataset1_array = dataset_class1.to_numpy()
dataset2_array = dataset_class2.to_numpy()
dataset3_array = dataset_class3.to_numpy()


data1_minus_f3 = np.delete(dataset1_array, 2, axis=1)
data2_minus_f3 = np.delete(dataset2_array, 2, axis=1)
data3_minus_f3 = np.delete(dataset3_array, 2, axis=1)

data1_minus_f2f3 = np.delete(data1_minus_f3, 1, axis=1)
data2_minus_f2f3 = np.delete(data2_minus_f3, 1, axis=1)
data3_minus_f2f3 = np.delete(data3_minus_f3, 1, axis=1)

#dataset1f3_train30 = data1_minus_f3[0:30]
#dataset2f3_train30 = data2_minus_f3[0:30]
#dataset3f3_train30 = data3_minus_f3[0:30]

dataset1f2f3_train30 = data1_minus_f2f3[0:30]
dataset2f2f3_train30 = data2_minus_f2f3[0:30]
dataset3f2f3_train30 = data3_minus_f2f3[0:30]

#dataset1f3_test20 = data1_minus_f3[30:]
#dataset2f3_test20 = data2_minus_f3[30:]
#dataset3f3_test20 = data3_minus_f3[30:]

dataset1f2f3_test20 = data1_minus_f2f3[30:]
dataset2f2f3_test20 = data2_minus_f2f3[30:]
dataset3f2f3_test20 = data3_minus_f2f3[30:]

#training and test datasets
#training_dataset_array = np.concatenate((dataset1f3_train30, dataset2f3_train30, dataset3f3_train30))
#test_dataset_array = np.concatenate((dataset1f3_test20, dataset2f3_test20, dataset3f3_test20))

training_dataset_arrayf2f3 = np.concatenate((dataset1f2f3_train30, dataset2f2f3_train30, dataset3f2f3_train30))
test_dataset_arrayf2f3 = np.concatenate((dataset1f2f3_test20, dataset2f2f3_test20, dataset3f2f3_test20))

#array with true values of 0,1,2 representing different flowers
#training_true_values = np.array(len(dataset1f3_train30)*[0]+len(dataset2f3_train30)*[1]+ len(dataset3f3_train30)*[2])

training_true_valuesf2f3 = np.array(len(dataset1f2f3_train30)*[0]+len(dataset2f2f3_train30)*[1]+ len(dataset3f2f3_train30)*[2])

##CHANGE VALUES
train_correct = np.zeros((training_true_valuesf2f3.shape[0],3))
for i, label in np.ndenumerate(training_true_valuesf2f3):
    train_correct[i][label] = 1


#do the same for test array
#test_true_values = np.array(len(dataset1f3_test20)*[0]+len(dataset2f3_test20)*[1]+ len(dataset3f3_test20)*[2])
test_true_valuesf2f3 = np.array(len(dataset1f2f3_test20)*[0]+len(dataset2f2f3_test20)*[1]+ len(dataset3f2f3_test20)*[2])


test_correct = np.zeros((test_true_valuesf2f3.shape[0],3))
for i, label in np.ndenumerate(test_true_valuesf2f3):
    test_correct[i][label] = 1


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
#CHANGE INPUT
W = training(training_dataset_arrayf2f3, train_correct, 0.007, 50000)

def predict(x, W):
    n = x.shape[0]
    prediction = np.array(
        [1.0/(1.0+np.exp(-(np.matmul(W,x[i])))) for i in range(n)])
    return prediction

#CHANGE INPUT
predicted_dataset = predict(training_dataset_arrayf2f3, W)

def generate_confusion_matrix(x, y, W):
    pred = predict(x, W)
    confusion_matrix = np.zeros((3, 3))

    for i in range(pred.shape[0]):
        confusion_matrix[np.argmax(y[i])][np.argmax(pred[i])] += 1
    return confusion_matrix

#print(generate_confusion_matrix(training_dataset_array, train_correct, W))
plt.figure(figsize = (7,7))
plt.title("Confusion matrix: Training set")
sn.heatmap(generate_confusion_matrix(training_dataset_arrayf2f3, train_correct, W), annot=True)

plt.figure(figsize = (7,7))
plt.title("Confusion matrix: Test set")
sn.heatmap(generate_confusion_matrix(test_dataset_arrayf2f3, test_correct, W), annot=True)

dataset_class1_mf3 = dataset_class1.drop(columns="p_length")
dataset_class2_mf3 = dataset_class2.drop(columns="p_length")
dataset_class3_mf3 = dataset_class3.drop(columns="p_length")

dataset_class1_mf2f3 = dataset_class1_mf3.drop(columns="s_width")
dataset_class2_mf2f3 = dataset_class2_mf3.drop(columns="s_width")
dataset_class3_mf2f3 = dataset_class3_mf3.drop(columns="s_width")

#dataset_class1_mf3.plot.hist(bins=50)
#dataset_class1_mf3.plot.hist(bins=50, alpha=0.7)
#dataset_class2_mf3.plot.hist(bins=50)
#dataset_class2_mf3.plot.hist(bins=50, alpha=0.7)
#dataset_class3_mf3.plot.hist(bins=50)
#dataset_class3_mf3.plot.hist(bins=50, alpha=0.7)
plt.show()
