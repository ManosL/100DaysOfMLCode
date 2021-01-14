import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

import hdf5storage

from isolationForest import IsolationForest
import sklearn.ensemble as ensemble
from sklearn.metrics import roc_auc_score

http_dataset = hdf5storage.loadmat('http.mat')

http_X       = http_dataset['X']
http_labels  = http_dataset['y']

sk_forest = ensemble.IsolationForest(100, 256)
sk_forest.fit(http_X)
preds = sk_forest.predict(http_X[0:1000000, :])

preds = [0 if pred == 1 else 1 for pred in preds]

print('Sklearn AUC Score:', roc_auc_score(http_labels[0:1000000], preds))

i_forest = IsolationForest(100, 256)
i_forest.fit(http_X)
preds    = i_forest.predict(http_X[0:1000000, :])

# If you want to categorize points as outliers and non-outliers
for i in range(len(preds)):
    if preds[i] >= 0.6:
        preds[i] = 1
    else:
        preds[i] = 0

print('My implementation AUC Score:', roc_auc_score(http_labels[0:1000000], preds))