# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 09:47:25 2019

@author: nghp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier

##### Import dataset and set it fruit
fruits = pd.read_table(r"H:\Python proj\Applied Machine Learning in Python\Lesson 1\fruit_data_with_colors.txt")

##### Look at the first few entries of the dataset
fruits.head()

#### Look out for fruit names 
lookup_Fruit_names = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))

#### Split Train Test set 
X = fruits[['mass', 'width', 'height','color_score']]
y = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0) 

#### Examine the data
cmap = cm.get_cmap('gnuplot')
scatter = pd.plotting.scatter_matrix(X_train, c = y_train, marker = 'o', s = 40,hist_kwds = {'bins':15}, cmap = cmap)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s = 100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
plt.show()

#### k-NN classifier
# 1) Create classifier object
knn = KNeighborsClassifier(n_neighbors = 5) 
# 2) Train the classifier 
knn.fit(X_train, y_train)
# 3) Estimate the accuracy of the classifier 
print("Accuracy of the K-NN classifier is:" ,knn.score(X_test, y_test))

#### Use the classifier to predict new, unseen data
example_fruit = [[20,4.3,5.5, 4.2]]
fruits_predicted = knn.predict(example_fruit)
print("The predicted fruit is:" , lookup_Fruit_names[fruits_predicted[0]])

