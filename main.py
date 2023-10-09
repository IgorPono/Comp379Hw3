import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
from sklearn.dummy import DummyClassifier


def accuracy(predictionArr, successes):  # X is the predictions, successes is the groundtruth
    correctPredictions = 0
    totalPredictions = 0
    for i in range(len(predictionArr)):
        totalPredictions = totalPredictions + 1
        if predictionArr[i] == successes[i]:
            correctPredictions = correctPredictions + 1
    return correctPredictions / totalPredictions


def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN(object):
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute distance
        distances = [distance(x, x_train) for x_train in self.X_train]
        # get closest k values
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]


df = pd.read_csv('train.csv', usecols=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
df = df.fillna(0)
df['Sex'].replace(['female', 'male'], [0, 1], inplace=True)


train, test = train_test_split(df, test_size=0.3)  # split off 30 percent of data into test
development, test = train_test_split(test, test_size=0.5)  # split test in half

survivedTrain = train['Survived'].tolist()
survivedDevelopment = development['Survived'].tolist()
survivedTest = test['Survived'].tolist()

'''train['Sex'].replace(['female','male'],[0,1], inplace =True)
development['Sex'].replace(['female','male'],[0,1], inplace =True)
test['Sex'].replace(['female','male'],[0,1], inplace =True)'''

train = train.drop(['Survived'], axis=1).values
development = development.drop(['Survived'], axis=1).values
test = test.drop(['Survived'], axis=1).values

print('Predictions on development set')
print()

SVM_Model = SVC(gamma='auto', C=10000)
SVM_Model.fit(train, survivedTrain)

print(f'sklearn SVM Development set Accuracy - : {SVM_Model.score(development, survivedDevelopment):.3f}')
print(f'sklearn SVM Development set f1 score - : {f1_score(survivedDevelopment, SVM_Model.predict(development)):.3f}')

#print(np.sqrt(len(test)))
my_KNN = KNN(math.floor(np.sqrt(len(test)))) # n will be 11 in this situation
my_KNN.fit(train, survivedTrain)

print('KNN Development set  Accuracy - : ' + str(accuracy(my_KNN.predict(development),survivedDevelopment)))

# Dummy_Classifer = DummyClassifier(strategy="most_frequent")
Dummy_Classifer = DummyClassifier(strategy="stratified")
Dummy_Classifer.fit(train, survivedTrain)
# Dummy_Classifer.predict(development)

print(f'sklearn Dummy Classifier Development set  Accuracy - : {Dummy_Classifer.score(development, survivedDevelopment):.3f}')
print(f'sklearn SVM Development set f1 score - : {f1_score(survivedDevelopment, Dummy_Classifer.predict(development)):.3f}')
