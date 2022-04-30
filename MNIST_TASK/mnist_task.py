from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
from scipy.spatial import distance
import seaborn as sn


def plot_input_img(x, y, test_img, train_img, train_labels, test_labels):
    x_reshaped = np.reshape(test_img[x], (28, 28))
    y_reshaped = np.reshape(train_img[y], (28, 28))
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(x_reshaped, cmap='binary', vmin=0, vmax=255)
    plt.title("Test number: {}".format(test_labels[x]))
    plt.subplot(1, 2, 2)
    plt.imshow(y_reshaped, cmap='binary', vmin=0, vmax=255)
    plt.title("Classified as: {}".format(train_labels[y]))


def plotConfusionMatrix(confusion_matrix):
    plt.figure(figsize=(7, 7))
    plt.title("Confusion matrix")
    sn.heatmap(confusion_matrix, annot=True, fmt=".1f")


def nearestNeighborClassifier(train_images, train_labels, test_images, test_labels, numberOfTrainImages, numberOfTestImages):
    # Array consisting of [test_image, test_label, train_image_train_label]
    predicted_images = []
    # For each test image in test images
    for test_image in range(numberOfTestImages):
        # Check it against every train_image in train_images
        dist = []
        for train_image in range(numberOfTrainImages):
            # Take the euclidian distance
            euclidian_distance = distance.euclidean(
                test_images[test_image], train_images[train_image])
            dist.append([euclidian_distance, train_image])
        # Sort the dist array to get the smallest distance
        dist.sort()
        train_image_number = dist[0][1]
        predicted_images.append(
            [test_image, test_labels[test_image], train_image_number, train_labels[train_image_number]])
    return predicted_images


def makePredictions(predicted_images):
    correct_predictions = []
    wrong_predictions = []
    for predicted_image in range(len(predicted_images)):
        if predicted_images[predicted_image][1] == predicted_images[predicted_image][3]:
            correct_predictions.append(predicted_images[predicted_image])
        else:
            wrong_predictions.append(predicted_images[predicted_image])
    return correct_predictions, wrong_predictions


def makeConfusionMatrix(correct_predictions, wrong_predictions):
    confusion_matrix = np.zeros((10, 10), dtype=int)
    for prediction in range(len(correct_predictions)):
        confusion_matrix[correct_predictions[prediction][1]
                         ][correct_predictions[prediction][1]] += 1
    for j in range(len(wrong_predictions)):
        confusion_matrix[wrong_predictions[j][1]][wrong_predictions[j][3]] += 1
    return confusion_matrix


def kMeans(X, Y):
    kmeans = KMeans(n_clusters=64, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    # gets a value from 0-64?
    pred_y = kmeans.fit_predict(X)
    return pred_y


def divideInClasses(train_images):
    n = 6000
    output = [train_images[i:i + n] for i in range(0, len(train_images), n)]
    return output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7], output[8], output[9]


def main():
    Ntrain = 60000
    Ntest = 10000
    # Opening the bin files
    with open('MNIST_TASK/train_images.bin', 'rb') as binaryFile:
        train_images = binaryFile.read()
    with open('MNIST_TASK/train_labels.bin', 'rb') as binaryFile:
        train_labels = binaryFile.read()
    with open('MNIST_TASK/test_images.bin', 'rb') as binaryFile:
        test_images = binaryFile.read()
    with open('MNIST_TASK/test_labels.bin', 'rb') as binaryFile:
        test_labels = binaryFile.read()
    # Reshaping the data to make it easier to use
    train_images = np.reshape(np.frombuffer(
        train_images[16:16+784*Ntrain], dtype=np.uint8), (Ntrain, 784))
    train_labels = np.frombuffer(train_labels[8:Ntrain+8], dtype=np.uint8)
    test_images = np.reshape(np.frombuffer(
        test_images[16:16+784*Ntest], dtype=np.uint8), (Ntest, 784))
    test_labels = np.frombuffer(test_labels[8:Ntest+8], dtype=np.uint8)
    # ------------TASK 1------------
    # Predict images
    predicted_images = nearestNeighborClassifier(
        train_images, train_labels, test_images, test_labels, Ntrain, Ntest)
    # Add correct predictions to correct_predictions and wrong predictions to wrong_predictions
    correct_predictions, wrong_predictions = makePredictions(predicted_images)
    # Make confusion matric with correct and wrong predictions
    confusion_matrix = makeConfusionMatrix(
        correct_predictions, wrong_predictions)
    # Plot the confusion matric
    plotConfusionMatrix(confusion_matrix)

    # ------------------------TASK 2---------------------------
    # Divide training images into 10 classes
    class1, class2, class3, class4, class5, class6, class7, class8, class9, class10 = divideInClasses(
        train_images)
    # Divide training labels into 10 classes
    label1, label2, label3, label4, label5, label6, label7, label8, label9, label10 = divideInClasses(
        train_labels)
    # Cluster the data
    cluster1 = kMeans(class1, train_labels)
    #cluster2 = kMeans(class2, train_labels)
    #cluster3 = kMeans(class3, train_labels)
    #cluster4 = kMeans(class4, train_labels)
    #cluster5 = kMeans(class5, train_labels)
    #cluster6 = kMeans(class6, train_labels)
    #cluster7 = kMeans(class7, train_labels)
    #cluster8 = kMeans(class8, train_labels)
    #cluster9 = kMeans(class9, train_labels)
    #cluster10 = kMeans(class10, train_labels)

    predicted_images = nearestNeighborClassifier(
        cluster1, label1, test_images, test_labels, 64, 1000)
    correct_predictions, wrong_predictions = makePredictions(predicted_images)
    confusion_matrix = makeConfusionMatrix(
        correct_predictions, wrong_predictions)
    plt.show()

    return


if __name__ == "__main__":
    main()
