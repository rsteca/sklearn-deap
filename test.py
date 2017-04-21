from evolutionary_search import EvolutionaryAlgorithmSearchCV, maximize
import sklearn.datasets
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import unittest
import random

def func(x, y, m=1., z=False):
    return m * (np.exp(-(x**2 + y**2)) + float(z))

def readme(n_jobs=1):
    data = sklearn.datasets.load_digits()
    X = data["data"]
    y = data["target"]

    paramgrid = {"kernel": ["rbf"],
                 "C": np.logspace(-9, 9, num=25, base=10),
                 "gamma": np.logspace(-9, 9, num=25, base=10)}

    random.seed(1)

    cv = EvolutionaryAlgorithmSearchCV(estimator=SVC(),
                                       params=paramgrid,
                                       scoring="accuracy",
                                       cv=StratifiedKFold(n_splits=4),
                                       verbose=1,
                                       population_size=10,
                                       gene_mutation_prob=0.10,
                                       gene_crossover_prob=0.5,
                                       tournament_size=3,
                                       generations_number=5,
                                       n_jobs=n_jobs)
    cv.fit(X, y)
    return cv

class TestEvolutionarySearch(unittest.TestCase):

    def test_cv(self):
        def try_with_params(**kwargs):
            cv = readme(**kwargs)
            cv_results_ = cv.cv_results_
            print("CV Results:\n{}".format(cv_results_))
            self.assertIsNotNone(cv_results_, msg="cv_results is None.")
            self.assertNotEqual(cv_results_, {}, msg="cv_results is empty.")
            self.assertAlmostEqual(cv.best_score_, 1., delta=.05,
                msg="Did not find the best score. Returned: {}".format(cv.best_score_))

        try_with_params(n_jobs=1)
        try_with_params(n_jobs=4)

    def test_optimize(self):
        """ Simple hill climbing optimization with some twists. """

        param_grid = {'x': [-1., 0., 1.], 'y': [-1., 0., 1.], 'z': [True, False]}
        args = {'m': 1.}

        def try_with_params(**max_args):
            best_params, best_score, score_results = maximize(func, param_grid,
                                                args, verbose=True, **max_args)
            print("Score Results:\n{}".format(score_results))
            self.assertEqual(best_params, {'x': 0., 'y': 0., 'z': True})
            self.assertEqual(best_score, 2.)

        try_with_params(n_jobs=1)
        try_with_params(n_jobs=4)


if __name__ == "__main__":
    unittest.main()
