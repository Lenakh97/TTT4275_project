import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

names = ["s_length", "s_width", "p_length", "p_width"]
dataset_class1 = pd.read_csv("class_1", names=names)
dataset_class2 = pd.read_csv("class_2", names=names)
dataset_class3 = pd.read_csv("class_3", names=names)

dataset_class1.hist()
dataset_class2.hist()
dataset_class3.hist()
plt.figure(figsize = (7,7))
dataset_class1.plot.hist(bins=50)
dataset_class1.plot.hist(bins=50, alpha=0.7)
dataset_class2.plot.hist(bins=50)
dataset_class2.plot.hist(bins=50, alpha=0.7)
dataset_class3.plot.hist(bins=50)
dataset_class3.plot.hist(bins=50, alpha=0.7)
plt.show()
