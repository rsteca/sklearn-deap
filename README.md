# sklearn-deap
Use evolutionary algorithms instead of gridsearch in scikit-learn. 

It's implemented using deap library: https://github.com/deap/deap

Install
-------

To install the library just type the following on your shell:

    python setup.py install

Usage examples
--------------



Example of usage:

```python
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

grid = {
    'svc__C': [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 10, 100, 1000],
    'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'pca__n_components': [1, 2, 3, 4]
}
iris = datasets.load_iris()
X, y = iris.data, iris.target
pipeline = Pipeline(steps=[
    ('pca', PCA()),
    ('svc', SVC())
])
clf = EvolutionaryAlgorithmSearchCV(pipeline, grid, scoring=None, verbose=True, n_jobs=4, population_size=5)
clf.fit(X, y)
```

Output:

        --- Evolve in 240 possible combinations ---
        gen nevals  avg     min         max     
        0   5       0.84188 0.615919    0.966346
        1   4       0.95438 0.926282    0.972756
        2   3       0.971581    0.966346    0.973291
        3   4       0.969017    0.952457    0.973291
        4   2       0.973291    0.973291    0.973291
        5   2       0.965171    0.932692    0.973291
        6   4       0.973291    0.973291    0.973291
        7   0       0.973291    0.973291    0.973291
        8   4       0.973291    0.973291    0.973291
        9   2       0.971902    0.966346    0.973291
        10  0       0.973291    0.973291    0.973291
        Best individual is: {"pca__n_components": 3, "svc__kernel": "linear", "svc__C": 1.0}
        with fitness: (0.9732905982905983,)
