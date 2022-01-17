import numpy as np
import pandas as pd


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

