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
def arrange(dat):
	xFeatures = ['HomeAway', 'Opp', 'Season', 'Rest']
	yFeatures = ['PTS']
	X = dat[xFeatures]
	Y = dat[yFeatures]
	header = ((dat.columns).to_numpy()).tolist()	
	XFeaturesA = header[1:3] + header[4:] # all feature columns except points (target)

	xFeatures3 = xFeatures.copy(); xFeatures3 = xFeatures3 + ['PTS_avg3', 'PTS_sd3']
	xFeatures5 = xFeatures3.copy(); xFeatures5 = xFeatures5 + ['PTS_avg5', 'PTS_sd5']
	xFeatures10 = xFeatures5.copy(); xFeatures10 = xFeatures10 + ['PTS_avg10', 'PTS_sd10']
	xFeaturesC = xFeatures10.copy(); xFeaturesC = xFeaturesC + ['PTS_career_avg', 'PTS_career_sd']
	X3 = dat[xFeatures3]; X5 = dat[xFeatures5]; X10 = dat[xFeatures10]; XC = dat[xFeaturesC]; 
	XA = dat[XFeaturesA]
	xfList = [xFeatures, xFeatures3, xFeatures5, xFeatures10, xFeaturesC, XFeaturesA]
	xDataSets = [X, X3, X5, X10, XC, XA]

	return xfList, xDataSets, Y


xfList, xDataSets, Y = arrange(dat)
#models = []
#featureImportances = []
for i in range(0, len(xDataSets)):
	Xtrain, Xtest, Ytrain, Ytest = train_test_split(xDataSets[i], Y, test_size=0.2, random_state=123)
	h = tree.DecisionTreeRegressor()
	#models.append(h)
	h.fit(Xtrain, Ytrain)
	k = h.feature_importances_
	xFeatures = xfList[i]
	w = dict()
	for j in range(0, len(xFeatures)):
		w[xFeatures[j]] = k[j]
	FIsorted = sorted(w.items(), key=lambda p: p[1], reverse=True)
	#featureImportances.append(FIsorted)
	print("\nNum features used in training: " + str(h.n_features_in_))
	print("Feature importances: ")
	if (i < 5):
		print(FIsorted)
	else:
		print(FIsorted[0:12])

	preds = h.predict(Xtest)
	print("Mean Squared Error: " + str(mean_squared_error(Ytest, preds)))
	print("Mean Absolute Error: " + str(mean_absolute_error(Ytest, preds)))
	print("R2: " + str(r2_score(Ytest, preds)))

	




# Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1, random_state=123)
# h = tree.DecisionTreeRegressor()
# h.fit(Xtrain,Ytrain)
# print(xFeatures)

# print("Default")
# print(h.feature_importances_)

# rep = input("Default Decision Tree with 4 features: Enter new hyperparameters y/n\n")
# while (not "n" in rep):
# 	depth = int(input("Max Depth (int): "))
# 	if (depth == 0): depth = None
# 	minSamplesLeaf = float(input("Min Samples Leaf (float between 0 and 1): "))
# 	if (minSamplesLeaf == 0.0): minSamplesLeaf = 0
# 	elif (minSamplesLeaf == 1.0): minSamplesLeaf = 1
# 	modelCur = tree.DecisionTreeRegressor(max_depth=depth, min_samples_leaf=minSamplesLeaf)
# 	modelCur.fit(X,Y)
# 	preds = modelCur.predict(Xtest)

# 	print("Feature Importances for 'HomeAway', 'Opp', 'Season', 'Rest': " + str(modelCur.feature_importances_))
# 	print("Mean Squared Error: " + str(mean_squared_error(Ytest, preds)))
# 	print("Mean Absolute Error: " + str(mean_absolute_error(Ytest, preds)))
# 	rep = input("\nEnter new hyperparameters y/n\n")
