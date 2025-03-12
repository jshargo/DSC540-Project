import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import math
#import random
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
# https://scikit-learn.org/stable/
from sklearn import tree

dat = pd.read_csv("DSC540-Project/LukaDoncicCleaned.csv")
# X = dat.iloc[:,1:].to_numpy(); Y = dat.iloc[:,0].to_numpy()
# <=4 input features, 1<= output feature 
xFeatures = ['HomeAway', 'Opp', 'Season', 'Rest']
yFeatures = ['PTS']
X = dat[xFeatures]
Y = dat[yFeatures]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1, random_state=123)
h = tree.DecisionTreeRegressor()
h.fit(X,Y)
print(xFeatures)

print("Default")
print(h.feature_importances_)
preds = h.predict(Xtest)
print(mean_squared_error(Ytest, preds))

rep = input("Enter new hyperparameters y/n\n")
while (not "n" in rep):
	depth = int(input("Max Depth (int): "))
	if (depth == 0): depth = 'None'
	minSamplesLeaf = float(input("Min Samples Leaf (float between 0 and 1): "))
	if (minSamplesLeaf == 0.0): minSamplesLeaf = 0
	elif (minSamplesLeaf == 1.0): minSamplesLeaf = 1

	modelCur = tree.DecisionTreeRegressor(max_depth=depth, min_samples_leaf=minSamplesLeaf)
	modelCur.fit(X,Y)
	print("Feature Importances for 'HomeAway', 'Opp', 'Season', 'Rest': " + str(modelCur.feature_importances_))
	preds = modelCur.predict(Xtest)
	print("Mean Squared Error: " + str(mean_squared_error(Ytest, preds)))
	print("Mean Absolute Error: " + str(mean_absolute_error(Ytest, preds)))
	rep = input("\nEnter new hyperparameters y/n\n")
