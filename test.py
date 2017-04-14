from evolutionary_search import EvolutionaryAlgorithmSearchCV
import sklearn.datasets
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
import pandas as pd

data = sklearn.datasets.load_digits()
X = data["data"]
y = data["target"]
# make it a 2-class problem by only classifying the digit \"5\" vs the rest
y = np.array([1 if yy == 5 else 0 for yy in y])
X.shape, y.shape
paramgrid = {"kernel": ["rbf"],
             "C"     : np.logspace(-9, 9, num=5, base=10),
             "gamma" : np.logspace(-9, 9, num=5, base=10)}
cv = EvolutionaryAlgorithmSearchCV(estimator=SVC(),
                                   params=paramgrid,
                                       scoring="accuracy",
                                       cv=StratifiedKFold(y, n_folds=10),
                                       verbose=True,
                                       population_size=10,
                                       gene_mutation_prob=0.10,
                                       tournament_size=3,
                                       generations_number=10)
cv.fit(X, y)
print(pd.DataFrame(cv.cv_results_))