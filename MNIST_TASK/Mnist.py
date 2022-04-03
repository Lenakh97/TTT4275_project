import numpy as np
import operator
from operator import itemgetter
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from scipy.spatial import distance
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from math import sqrt

#Eucledian distance to check distance between two vectors
def euclidean_distance(image1, image2):
    distance = 0.0
    for i in range(len(image1)-1):
        distance += (image1[i] - image2[i])**2
    return sqrt(distance)

def nearestNeighborClassifier(x_train, x_test, y_train, y_test, num_neighbors):
    #for image in x_test
    for test_image in range(10):
        distances = list()
        for train_image in range(50):
            dist = np.linalg.norm(x_test[test_image]-x_train[train_image])
            distances.append((x_train[train_image], dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    print(prediction)
    return prediction

#Get the Nearest Neighbors
def getNearestNeighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        #calculates the distance between the test row and the train row
        #dist = euclidean_distance(test_row, train_row)
        dist = np.linalg.norm(test_row-train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    print("neigh", neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    print(prediction)
    return prediction

#Predicting classification with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = getNearestNeighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	#prediction = max(set(output_values), key=output_values.count)
	#return prediction

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def plot_input_img(i):
    plt.imshow(x_train[i], cmap = 'binary')
    plt.title(y_train[i])
    plt.show()

#for i in range(10):
    plot_input_img(i)

#preprocess our image

x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255


#expand the dimentions of images to (28,28,1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
#print(x_train) 

#convert classes to one hot vectors
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

nearestNeighborClassifier(x_train, x_test, y_train, y_test, 3)

#dataset = [[2.7810836,2.550537003,0],
#	[1.465489372,2.362125076,0],
#	[3.396561688,4.400293529,0],
#	[1.38807019,1.850220317,0],
#	[3.06407232,3.005305973,0],
#	[7.627531214,2.759262235,1],
#	[5.332441248,2.088626775,1],
#	[6.922596716,1.77106367,1],
#	[8.675418651,-0.242068655,1],
#	[7.673756466,3.508563011,1]]
#neighbors = getNearestNeighbors(dataset, dataset[0], 3)
#for neighbor in neighbors:
#	print(neighbor)
#print(x_train)
#prediction = predict_classification(dataset, dataset[0], 3)
#for test_row in x_test:
#    pretiction = predict_classification(x_train, test_row, 3)
#print('Expected %d, Got %d' % (dataset[0][-1], prediction))

print(float('inf'))
print(1<float('inf'))