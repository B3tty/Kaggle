# # data analysis and wrangling
import pandas as pd
# import numpy as np
# import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#matplotlib inline

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
    def __init__(self, combine):
        self.combine = combine
        train_df = combine[0]
        test_df = combine[1]
        self.X_train = train_df.drop("SalePrice", axis=1)
        self.Y_train = train_df["SalePrice"]
        self.X_test = test_df.drop("Id", axis=1).copy()

    def main(self):
        self.logistic_regression()
        self.svm()
        self.knn()
        self.gaussian_naive_bayes()
        self.perceptron()
        self.random_forest()

    def logistic_regression(self):
        print("logistic regression beginning")

        logreg = LogisticRegression()
        logreg.fit(self.X_train, self.Y_train)
        Y_pred = logreg.predict(self.X_test)

        coeff_df = pd.DataFrame(self.combine[0].columns.delete(0))
        coeff_df.columns = ['Feature']
        coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
        # print(coeff_df.sort_values(by='Correlation', ascending=False))

        acc_log = round(logreg.score(self.X_train, self.Y_train) * 100, 2)
        print(f"logistic regression done with {acc_log}% accuracy")
        return Y_pred

    def svm(self):
        print("support vector machines beginning")
        svc = SVC()
        svc.fit(self.X_train, self.Y_train)
        Y_pred = svc.predict(self.X_test)
        acc_svc = round(svc.score(self.X_train, self.Y_train) * 100, 2)
        print(f"support vector machines done with {acc_svc}% accuracy")
        return Y_pred

    def knn(self):
        print("knn beginning")
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(self.X_train, self.Y_train)
        Y_pred = knn.predict(self.X_test)
        acc_knn = round(knn.score(self.X_train, self.Y_train) * 100, 2)
        print(f"knn with {acc_knn}% accuracy")
        return Y_pred

    def gaussian_naive_bayes(self):
        print("gaussian naive bayes beginning")
        gaussian = GaussianNB()
        gaussian.fit(self.X_train, self.Y_train)
        Y_pred = gaussian.predict(self.X_test)
        acc_gaussian = round(gaussian.score(self.X_train, self.Y_train) * 100, 2)
        print(f"gaussian naive bayes done with {acc_gaussian}% accuracy")
        return Y_pred

    def perceptron(self):
        print("perceptron beginning")
        perceptron = Perceptron()
        perceptron.fit(self.X_train, self.Y_train)
        Y_pred = perceptron.predict(self.X_test)
        acc_perceptron = round(perceptron.score(self.X_train, self.Y_train) * 100, 2)
        print(f"perceptron done with {acc_perceptron}% accuracy")
        return Y_pred

    def random_forest(self):
        print("random forest beginning")
        random_forest = RandomForestClassifier(n_estimators=100)
        random_forest.fit(self.X_train, self.Y_train)
        Y_pred = random_forest.predict(self.X_test)
        random_forest.score(self.X_train, self.Y_train)
        acc_random_forest = round(random_forest.score(self.X_train, self.Y_train) * 100, 2)
        print(f"random forest done with {acc_random_forest}% accuracy")
        return Y_pred
