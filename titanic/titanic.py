# # data analysis and wrangling
import pandas as pd
# import numpy as np
# import random as rnd
#
# # visualization
# import seaborn as sns
# import matplotlib.pyplot as plt
# #matplotlib inline
#
# # machine learning
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import SGDClassifier
# from sklearn.tree import DecisionTreeClassifier

import features as ft
import models

# Acquire data
train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
combine = [train_df, test_df]

# Feature Engineering
feature_engineer = ft.FeatureEngineer()
combine = feature_engineer.engineer(combine)

# Modelisation
modeler = models.Model(combine)
prediction = modeler.random_forest()

# Output formatting
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": prediction
    })
submission.to_csv('output/submission.csv', index=False)