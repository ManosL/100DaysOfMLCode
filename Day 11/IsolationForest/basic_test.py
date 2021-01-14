import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

# Printing the original points
boston = load_boston()['data'][:, [5, 12]]
plt.scatter(boston[:, 0], boston[:, 1])
plt.show()

from isolationForest import IsolationForest

i_forest = IsolationForest(100, 256)
i_forest.fit(boston)
preds    = i_forest.predict(boston)

"""
# If you want to categorize points as outliers and non-outliers
for i in range(len(preds)):
    if preds[i] >= 0.6:
        preds[i] = 0
    else:
        preds[i] = 1
"""

# Printing the results of the isolation forest i wrote
plt.scatter(boston[:, 0], boston[:, 1], c=preds, cmap='winter')
plt.colorbar()
plt.show()

import sklearn.ensemble as ensemble

sk_forest = ensemble.IsolationForest(100, 256)
sk_forest.fit(boston)
preds = sk_forest.predict(boston)

# Printing the results of isolation forest of sklearn
plt.scatter(boston[:, 0], boston[:, 1], c=preds, cmap='winter')
plt.colorbar()
plt.show()