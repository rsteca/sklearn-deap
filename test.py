from evolutionary_search import EvolutionaryAlgorithmSearchCV, maximize
import sklearn.datasets
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import unittest

class TestEvolutionarySearch(unittest.TestCase):
    def test_cv(self):
        data = sklearn.datasets.load_digits()
        X = data["data"]
        y = data["target"]
        # make it a 2-class problem by only classifying the digit \"5\" vs the rest
        y = np.array([1 if yy == 5 else 0 for yy in y])
        X.shape, y.shape
        paramgrid = {"kernel": ["rbf"],
                     "C": np.logspace(-9, 9, num=5, base=10),
                     "gamma": np.logspace(-9, 9, num=5, base=10)}

        cv = EvolutionaryAlgorithmSearchCV(estimator=SVC(),
                                           params=paramgrid,
                                           scoring="accuracy",
                                           cv=StratifiedKFold(n_splits=10),
                                           verbose=False,
                                           population_size=10,
                                           gene_mutation_prob=0.10,
                                           tournament_size=3,
                                           generations_number=10)
        cv.fit(X, y)
        cv_results_ = cv.cv_results_
        self.assertIsNotNone(cv_results_, msg="cv_results is None.")
        self.assertNotEqual(cv_results_, {}, msg="cv_results is empty.")
        self.assertAlmostEqual(cv.best_score_, 1., delta=.05, msg="Did not find the best score. Returned: {}".format(cv.best_score_))

    def test_optimize(self):
        """ Simple hill climbing optimization with some twists. """
        def func(x, y, m=1., z=False):
            return m * (np.exp(-(x**2 + y**2)) + float(z))

        param_grid = {'x': [-1., 0., 1.], 'y': [-1., 0., 1.], 'z': [True, False]}
        args = {'m': 1.}
        best_params, best_score, _ = maximize(func, param_grid, args, verbose=False)
        self.assertEqual(best_params, {'x': 0., 'y': 0., 'z': True})
        self.assertEqual(best_score, 2.)


if __name__ == "__main__":
    unittest.main()
