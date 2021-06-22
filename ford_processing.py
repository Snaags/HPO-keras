
"""
Title: Timeseries classification from scratch
Author: [hfawaz](https://github.com/hfawaz/)
Date created: 2020/07/21
Last modified: 2020/08/21
Description: Training a timeseries classifier from scratch on the FordA dataset from the UCR/UEA archive.
"""
"""
## Introduction

This example shows how to do timeseries classification from scratch, starting from raw
CSV timeseries files on disk. We demonstrate the workflow on the FordA dataset from the
[UCR/UEA archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).

"""

"""
## Setup

"""

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"""
## Load the data: the FordA dataset

### Dataset description

The dataset we are using here is called FordA.
The data comes from the UCR archive.
The dataset contains 3601 training instances and another 1320 testing instances.
Each timeseries corresponds to a measurement of engine noise captured by a motor sensor.
For this task, the goal is to automatically detect the presence of a specific issue with
the engine. The problem is a balanced binary classification task. The full description of
this dataset can be found [here](http://www.j-wichard.de/publications/FordPaper.pdf).

### Read the TSV data

We will use the `FordA_TRAIN` file for training and the
`FordA_TEST` file for testing. The simplicity of this dataset
allows us to demonstrate effectively how to use ConvNets for timeseries classification.
In this file, the first column corresponds to the label.
"""
from scipy.io import arff
def readucr(filename):
    data = arff.loadarff(filename)
    df = pd.DataFrame(data[0])
    y = df["target"]
    X = df.loc[:, df.columns != 'target']
    y = np.asarray(y)
    X = np.asarray(X)

    return X, y.astype(int)


root_url = "/home/snaags/scripts/HPO-keras/datasets/Ford/"

x_train, y_train = readucr(root_url + "FordB_TRAIN.arff")
x_test, y_test = readucr(root_url + "FordB_TEST.arff")



"""
## Visualize the data

Here we visualize one timeseries example for each class in the dataset.

"""

classes = np.unique(np.concatenate((y_train, y_test), axis=0))

plt.figure()
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label="class " + str(c))
plt.legend(loc="best")
plt.show()
plt.close()

"""
## Standardize the data

Our timeseries are already in a single length (176). However, their values are
usually in various ranges. This is not ideal for a neural network;
in general we should seek to make the input values normalized.
For this specific dataset, the data is already z-normalized: each timeseries sample
has a mean equal to zero and a standard deviation equal to one. This type of
normalization is very common for timeseries classification problems, see
[Bagnall et al. (2016)](https://link.springer.com/article/10.1007/s10618-016-0483-9).

Note that the timeseries data used here are univariate, meaning we only have one channel
per timeseries example.
We will therefore transform the timeseries into a multivariate one with one channel
using a simple reshaping via numpy.
This will allow us to construct a model that is easily applicable to multivariate time
series.
"""


"""
Finally, in order to use `sparse_categorical_crossentropy`, we will have to count
the number of classes beforehand.
"""

num_classes = len(np.unique(y_train))

"""
Now we shuffle the training set because we will be using the `validation_split` option
later when training.
"""

"""
Standardize the labels to positive integers.
The expected labels will then be 0 and 1.
"""

y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

np.save("/home/snaags/scripts/HPO-keras/datasets/Ford/processedb/x_train.npy",x_train)
np.save("/home/snaags/scripts/HPO-keras/datasets/Ford/processedb/y_train.npy",y_train)
np.save("/home/snaags/scripts/HPO-keras/datasets/Ford/processedb/x_test.npy",x_test)
np.save("/home/snaags/scripts/HPO-keras/datasets/Ford/processedb/y_test.npy",y_test)
exit()

