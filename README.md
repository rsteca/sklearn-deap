# sklearn-deap
Use evolutionary algorithms instead of gridsearch in scikit-learn. This allows you to exponentially reduce the time required to find the best parameters for your estimator. Instead of trying out every possible combination of parameters, evolve only the combinations that give the best results.

[Here](https://github.com/rsteca/sklearn-deap/blob/master/notebooks/test.ipynb) is an ipython notebook comparing EvolutionaryAlgorithmSearchCV against GridSearchCV and RandomizedSearchCV.

It's implemented using deap library: https://github.com/deap/deap

Install
-------

To install the library just type the following on your shell:

    python setup.py install

Usage examples
--------------



Example of usage:

```python
import sklearn.datasets
import numpy as np
import random

data = sklearn.datasets.load_digits()
X = data["data"]
y = data["target"]

from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold

paramgrid = {"kernel": ["rbf"],
             "C"     : np.logspace(-9, 9, num=25, base=10),
             "gamma" : np.logspace(-9, 9, num=25, base=10)}

random.seed(1)

from evolutionary_search import EvolutionaryAlgorithmSearchCV
cv = EvolutionaryAlgorithmSearchCV(estimator=SVC(),
                                   params=paramgrid,
                                   scoring="accuracy",
                                   cv=StratifiedKFold(y, n_folds=4),
                                   verbose=1,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=5,
                                   n_jobs=4)
cv.fit(X, y)
```

Output:

        Types [1, 2, 2] and maxint [0, 24, 24] detected
        --- Evolve in 625 possible combinations ---
        gen	nevals	avg     	min    	max     
        0  	50    	0.202404	0.10128	0.962716
        1  	26    	0.383083	0.10128	0.962716
        2  	31    	0.575214	0.155259	0.962716
        3  	29    	0.758308	0.105732	0.976071
        4  	22    	0.938086	0.158041	0.976071
        5  	26    	0.934201	0.155259	0.976071
        Best individual is: {'kernel': 'rbf', 'C': 31622.776601683792, 'gamma': 0.001}
        with fitness: 0.976071229827
