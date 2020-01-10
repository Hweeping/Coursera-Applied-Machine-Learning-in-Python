# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 10:00:33 2020

@author: nghp
"""

from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


np.set_printoptions(precision=2)


fruits = pd.read_table('H:\\Python proj\\Applied Machine Learning in Python\\Lesson 2\\fruit_data_with_colors.txt')

feature_names_fruits = ['height', 'width', 'mass', 'color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']


clf = KNeighborsClassifier(n_neighbors = 5)
X = X_fruits_2d.as_matrix()
y = y_fruits_2d.as_matrix()
cv_scores = cross_val_score(clf, X, y)

print('Cross-validation scores (3-fold):', cv_scores)
print('Mean cross-validation score (3-fold): {:.3f}'
     .format(np.mean(cv_scores)))