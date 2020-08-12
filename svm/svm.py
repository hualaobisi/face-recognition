
import numpy as np

import logging

from matplotlib import pyplot as plt

from time import time

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# loading dataset
logging.basicConfig(level = logging.INFO, format="%(asctime)s %(message)s")
data_home = 'code/datasets/'
logging.info("start loading data")
faces = fetch_olivetti_faces(data_home=data_home)
logging.info("Complete")

# label
X = faces.data
y = faces.target
targets = np.unique(faces.target)
target_names = np.array(["p%d" % t for t in targets])
n_targets = target_names.shape[0]
n_samples, h, w = faces.images.shape

# show image
def plot_gallery(images, titles, h, w, n_row=2, n_col=5):

    plt.figure(figsize=(2*n_col, 2.2*n_row),dpi=140)
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.01)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(images[i].reshape((h,w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.axis('off')


sample_images = None
sample_titles = []
for i in range(n_targets):
    people_images = X[y == i]
    people_sample_index = np.random.randint(0, people_images.shape[0], 1)
    people_sample_image = people_images[people_sample_index, :]
    if sample_images is not None:
        sample_images = np.concatenate((sample_images, people_sample_image), axis=0)
    else:
        sample_images = people_sample_image
    sample_titles.append(target_names[i])

# loading data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


# svm
t = time()
clf = SVC(C=1.0, class_weight='balanced', coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(X_train, y_train)
print("time：{}s".format(time() - t))

# classification report
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=range(n_targets))
print("confusion_matrix:\n")
np.set_printoptions(threshold=np.inf)
print(cm[:])



