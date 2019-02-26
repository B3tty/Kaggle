# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#matplotlib inline
import cufflinks as cf

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


class Model(object):
    def logistic_regression(self, combine):
        print("logistic regression beginning")
        train_df = combine[0]
        test_df = combine[1]
        X_train = train_df.drop("Survived", axis=1)
        Y_train = train_df["Survived"]
        X_test = test_df.drop("PassengerId", axis=1).copy()

        print(X_train.shape, Y_train.shape, X_test.shape)

