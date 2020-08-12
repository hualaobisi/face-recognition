
import numpy as np

import logging

from matplotlib import pyplot as plt
from time import time
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# loading dataset
logging.basicConfig(level = logging.INFO, format="%(asctime)s %(message)s")

data_home = 'code/datasets/'
logging.info("start loading data")
faces = fetch_olivetti_faces(data_home=data_home)
logging.info("loading complete")

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

# find k value
candidate_components = range(10, 300, 10)
explained_ratios = []

for c in candidate_components:
    pca = PCA(n_components=c)
    X_pca = pca.fit_transform(X)
    explained_ratios.append(np.sum(pca.explained_variance_ratio_))

# plot k value
plt.figure(figsize=(8, 5), dpi=100)
plt.grid()
plt.plot(candidate_components, explained_ratios)
plt.xlabel('Number of PCA Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained variance ratio for PCA')
plt.yticks(np.arange(0.5, 1.05, .05))
plt.xticks(np.arange(0, 300, 20));

# show different k value image
def title_prefix(prefix, title):
    return "{}: {}".format(prefix, title)

n_row = 1
n_col = 5

sample_images = sample_images[0:5]
sample_titles = sample_titles[0:5]

plotting_images = sample_images
plotting_titles = [title_prefix('orig', t) for t in sample_titles]
candidate_components = [120, 80, 40, 10]
for c in candidate_components:

    pca = PCA(n_components=c)
    pca.fit(X)
    X_sample_pca = pca.transform(sample_images)
    X_sample_inv = pca.inverse_transform(X_sample_pca)
    plotting_images = np.concatenate((plotting_images, X_sample_inv), axis=0)
    sample_title_pca = [title_prefix('{}'.format(c), t) for t in sample_titles]
    plotting_titles = np.concatenate((plotting_titles, sample_title_pca), axis=0)

plot_gallery(plotting_images, plotting_titles, h, w,
    n_row * (len(candidate_components) + 1), n_col)
plt.show()

# choose k=120
n_components = 120

# pca
t = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Searching the best parameters for SVC
param_grid = {'C': [1, 5, 10, 50],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01]}
clf = GridSearchCV(SVC(class_weight='balanced', coef0=0.0,
    decision_function_shape=None, degree=3, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False), param_grid, verbose=2, n_jobs=4)
clf = clf.fit(X_train_pca, y_train)
print(clf.best_params_)
print("time：{}s".format(time() - t))

# classification report
y_pred = clf.best_estimator_.predict(X_test_pca)
print(classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=range(n_targets))
print("confusion matrix:")
np.set_printoptions(threshold=np.inf)
print(cm[:])


