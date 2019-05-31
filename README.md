# sklearn-deap
Use evolutionary algorithms instead of gridsearch in scikit-learn. This allows you to reduce the time required to find the best parameters for your estimator. Instead of trying out every possible combination of parameters, evolve only the combinations that give the best results.

[Here](https://github.com/rsteca/sklearn-deap/blob/master/test.ipynb) is an ipython notebook comparing EvolutionaryAlgorithmSearchCV against GridSearchCV and RandomizedSearchCV.

It's implemented using deap library: https://github.com/deap/deap

Install
-------

To install the library use pip:

    pip install sklearn-deap


or clone the repo and just type the following on your shell:

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
from sklearn.model_selection import StratifiedKFold

paramgrid = {"kernel": ["rbf"],
             "C"     : np.logspace(-9, 9, num=25, base=10),
             "gamma" : np.logspace(-9, 9, num=25, base=10)}

random.seed(1)

from evolutionary_search import EvolutionaryAlgorithmSearchCV
cv = EvolutionaryAlgorithmSearchCV(estimator=SVC(),
                                   params=paramgrid,
                                   scoring="accuracy",
                                   cv=StratifiedKFold(n_splits=4),
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

Example for maximizing just some function:

```python
from evolutionary_search import maximize

def func(x, y, m=1., z=False):
    return m * (np.exp(-(x**2 + y**2)) + float(z))

param_grid = {'x': [-1., 0., 1.], 'y': [-1., 0., 1.], 'z': [True, False]}
args = {'m': 1.}
best_params, best_score, score_results, _, _ = maximize(func, param_grid, args, verbose=False)
```

Output:

```python
best_params = {'x': 0.0, 'y': 0.0, 'z': True}
best_score  = 2.0
score_results = (({'x': 1.0, 'y': -1.0, 'z': True}, 1.1353352832366128),
 ({'x': -1.0, 'y': 1.0, 'z': True}, 1.3678794411714423),
 ({'x': 0.0, 'y': 1.0, 'z': True}, 1.3678794411714423),
 ({'x': -1.0, 'y': 0.0, 'z': True}, 1.3678794411714423),
 ({'x': 1.0, 'y': 1.0, 'z': True}, 1.1353352832366128),
 ({'x': 0.0, 'y': 0.0, 'z': False}, 2.0),
 ({'x': -1.0, 'y': -1.0, 'z': False}, 0.36787944117144233),
 ({'x': 1.0, 'y': 0.0, 'z': True}, 1.3678794411714423),
 ({'x': -1.0, 'y': -1.0, 'z': True}, 1.3678794411714423),
 ({'x': 0.0, 'y': -1.0, 'z': False}, 1.3678794411714423),
 ({'x': 1.0, 'y': -1.0, 'z': False}, 1.1353352832366128),
 ({'x': 0.0, 'y': 0.0, 'z': True}, 2.0),
 ({'x': 0.0, 'y': -1.0, 'z': True}, 2.0))
```

Using Keras model 

```python
from evolutionary_search import EvolutionaryAlgorithmSearchCV
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

SEED = 13
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

#...
# LOAD YOUR DATA, PREPROCESS AND SPLIT IT
# ...

def build_model(n_hidden=1, n_neurons=30, learning_rate=0.05, input_shape=[9], activation="relu"):
    model = keras.models.Sequential()
    options = {"input_shape": input_shape}
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(
            n_neurons, activation=activation, **options))
        options = {}

    model.add(keras.layers.Dense(13, activation="softmax", **options))
    s = 20 * len(X_train) // 32
    learning_rate_fn = keras.optimizers.schedules.ExponentialDecay(
        learning_rate, s, 0.1)
    optimizer = keras.optimizers.SGD(learning_rate_fn)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])
    return model

# Create a keras wrapper for scikit-sklearn using the build_model function
model = keras.wrappers.scikit_learn.KerasClassifier(build_model)

# Extra params for fit method must be passed to __init__ in recent version of scikit-sklearn
fit_params = {
    "epochs": 300,
    "validation_data": (X_valid, y_valid),
    "callbacks": [keras.callbacks.EarlyStopping(patience=10),
                keras.callbacks.ModelCheckpoint(model_filename,
                                                save_best_only=True)]
    }
params = {
        "n_hidden": [2, 3],
        "n_neurons": np.arange(10, 500),
        "learning_rate": [0.001, 0.025, 0.03]
    }
search = EvolutionaryAlgorithmSearchCV(estimator=model,
                                        params=params,
                                        scoring="accuracy",
                                        cv=cv,
                                        verbose=0,
                                        population_size=50,
                                        gene_mutation_prob=0.10,
                                        gene_crossover_prob=0.5,
                                        tournament_size=3,
                                        generations_number=5,
                                        fit_params=fit_params)

search.fit(X_train, y_train)
print(f"Best score found is: {search.best_score_}")
print(f"Best params are: {search.best_params_}")
```