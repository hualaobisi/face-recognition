from __future__ import division

import logging
import time

from matplotlib import pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
logging.basicConfig(level = logging.INFO, format="%(asctime)s %(message)s")

data_home = 'code/datasets/'
logging.info("Start loading data")
faces = fetch_olivetti_faces(data_home=data_home)
logging.info("Loading completed")

import  numpy as np

X = faces.data
y = faces.target

targets = np.unique(faces.target)
target_names = np.array(["p%d" % t for t in targets])
n_targets = target_names.shape[0]
n_samples, h, w = faces.images.shape

print('Samples count:{}\nTarget count:{}'.format(n_samples, n_targets))
print('Image size:{}x{}\nData shape:{}'.format(w, h, X.shape))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)

def distance(X_test, X_train):
    num_test = X_test.shape[0]
    num_train = X_train.shape[0]
    distances = np.zeros((num_test, num_train))
    dist1 = np.multiply(np.dot(X_test, X_train.T), -2)
    dist2 = np.sum(np.square(X_test), axis=1, keepdims=True)
    dist3 = np.sum(np.square(X_train.T), axis=0, keepdims=True)
    distances = np.sqrt(dist1 + dist2 + dist3)

    return distances


def predict(X_test, X_train, Y_train, k):

    distances = distance(X_test, X_train)
    num_test = X_test.shape[0]
    Y_prediction = np.zeros(num_test)
    for i in range(num_test):
        dists_min_k = np.argsort(distances[i])[:k]
        y_labels_k = Y_train[dists_min_k]
        Y_prediction[i] = np.argmax(np.bincount(y_labels_k))

    return Y_prediction, distances


def model(X_test, Y_test, X_train, Y_train, k , print_correct = False):

    Y_prediction, distances = predict(X_test, X_train, Y_train, k)
    num_correct = np.sum(Y_prediction == Y_test)
    accuracy = np.mean(Y_prediction == Y_test)
    if print_correct:
        print('Correct %d/%d: The test accuracy: %f' % (num_correct, X_test.shape[1], accuracy))
    d = {"k": k,
        "Y_prediction": Y_prediction,
        "distances" : distances,
        "accuracy": accuracy}
    return d

models = {}
for k in [1, 3, 5, 10]:
    start3=time.time()
    print ("k = " + str(k))
    models[str(k)] = model(X_test, y_test, X_train, y_train, k, print_correct = True)
    end3 = time.time()
    timecost3 = end3-start3
    print('tiecost=',timecost3)
    print ('\n' + "-------------------------------------------------------" + '\n')
models = {}
k = []
accuracys = []
for i in range(1,11):
    models[str(i)] = model(X_test, y_test, X_train, y_train, i, print_correct = False)
    k.append(models[str(i)]["k"])
    accuracys.append(models[str(i)]["accuracy"])
plt.grid()
plt.plot(k, accuracys)
plt.ylabel('accuracy')
plt.xlabel('k')
plt.xticks(np.arange(1, 10, 1));
plt.show()