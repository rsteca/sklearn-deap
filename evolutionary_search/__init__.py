# -*- coding: utf-8 -*-
import numpy as np
import random
from deap import base, creator, tools, algorithms
from multiprocessing import Pool
from sklearn.base import clone, is_classifier
try:
    from sklearn.model_selection import check_cv
    from sklearn.model_selection._search import BaseSearchCV
    from sklearn.model_selection._validation import _fit_and_score
except ImportError:
    from sklearn.cross_validation import check_cv, _fit_and_score
    from sklearn.grid_search import BaseSearchCV
from sklearn.metrics.scorer import check_scoring
from sklearn.utils.validation import _num_samples, indexable


def enum(**enums):
    return type('Enum', (), enums)

param_types = enum(Categorical=1, Numerical=2)


def _get_param_types_maxint(params):
    """
    Returns characteristics of parameters
    :param params: dictionary of pairs
        it must have parameter_name:list of possible values:
        params = {"kernel": ["rbf"],
                 "C"     : [1,2,3,4,5,6,7,8],
                 "gamma" : np.logspace(-9, 9, num=25, base=10)}
    :return: name_values pairs - list of (name,possible_values) tuples for each parameter
             types - list of types for each parameter
             maxints - list of maximum integer for each particular gene in chromosome
    """
    name_values = list(params.items())
    types = []
    for _, possible_values in name_values:
        if isinstance(possible_values[0], float):
            types.append(param_types.Numerical)
        else:
            types.append(param_types.Categorical)
    maxints = [len(possible_values) - 1 for _, possible_values in name_values]
    return name_values, types, maxints


def _initIndividual(pcls, maxints):
    part = pcls(random.randint(0, maxint) for maxint in maxints)
    return part


def _mutIndividual(individual, up, indpb, gene_type=None):
    for i, up, rn in zip(range(len(up)), up, [random.random() for _ in range(len(up))]):
        if rn < indpb:
            individual[i] = random.randint(0, up)
    return individual,


def _cxIndividual(ind1, ind2, indpb, gene_type):
    for i, gt, rn in zip(range(len(ind1)), gene_type, [random.random() for _ in range(len(ind1))]):
        if rn > indpb:
            continue
        if gt is param_types.Categorical:
            ind1[i], ind2[i] = ind2[i], ind1[i]
        else:
            # Case when parameters are numerical
            if ind1[i] <= ind2[i]:
                ind1[i] = random.randint(ind1[i], ind2[i])
                ind2[i] = random.randint(ind1[i], ind2[i])
            else:
                ind1[i] = random.randint(ind2[i], ind1[i])
                ind2[i] = random.randint(ind2[i], ind1[i])

    return ind1, ind2


def _individual_to_params(individual, name_values):
    return dict((name, values[gene]) for gene, (name, values) in zip(individual, name_values))


def _evalFunction(individual, searchobj, name_values, X, y, scorer, cv, iid, fit_params,
                  verbose=0, error_score='raise'):
    parameters = _individual_to_params(individual, name_values)
    score = 0
    n_test = 0
    for train, test in cv:
        paramkey = str(parameters)
        if paramkey in searchobj.score_cache:
            searchobj.num_cache_hits += 1
            _score = searchobj.score_cache[paramkey]
        else:
            _score, _, _ = _fit_and_score(estimator=individual.est, X=X, y=y, scorer=scorer,
                                     train=train, test=test, verbose=verbose,
                                     parameters=parameters, fit_params=fit_params,
                                     error_score=error_score)
            searchobj.num_evaluations += 1
            searchobj.score_cache[paramkey] = _score
        if searchobj.verbose and (searchobj.num_evaluations + searchobj.num_cache_hits) % searchobj.population_size == 0:
            print("Scoring evaluations: %d, Cache hits: %d, Total: %d" % (
                searchobj.num_evaluations, searchobj.num_cache_hits, searchobj.num_evaluations + searchobj.num_cache_hits))
        if iid:
            score += _score * len(test)
            n_test += len(test)
        else:
            score += _score
            n_test += 1
    score /= float(n_test)

    return (score,)


class EvolutionaryAlgorithmSearchCV(BaseSearchCV):
    """Evolutionary search of best hyperparameters, based on Genetic
    Algorithms

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    params : dict or list of dictionaries
        each dictionary must have parameter_name:sorted(list of possible values):
        params = {"kernel": ["rbf"],
                 "C"     : [1,2,3,4,5,6,7,8],
                 "gamma" : np.logspace(-9, 9, num=25, base=10)}
        Notice that Numerical values (floats) must be ordered in ascending or descending
        order.

    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    population_size : int, default=50
        Population size of genetic algorithm

    gene_mutation_prob : float, default=0.1
        Probability of gene mutation in chromosome

    gene_crossover_prob : float, default=0.5
        Probability of gene swap between two chromosomes

    tournament_size : int, default=3
        Size of tournament for selection stage of genetich algorithm

    generations_number : int, default=10
        Number of generations

    gene_type : list of integers, default=None
        list of types for each parameter, if None - it gets inferred from
        params, if some parameter has list of float values - it becomes param_types.Numerical
        if it has any other type of values - it becomes param_types.Categorical

        For Categorical features crossover operation just swaps corresponding
        genes between chromosomes with probability 'gene_crossover_prob'.

        For Numerical features crossover operation picks random gene between two
        genes of parents. Thus offsprings will have value of particular Numerical
        parameter in range [ind1_parameter, ind2_parameter]. Of course it is correct only
        when parameters of some value is sorted.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

        Currently it's not working

    pre_dispatch : int, or string, optional
        Dummy parameter for compatibility with sklearn's GridSearch

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this GridSearchCV instance after fitting.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.


    Examples
    --------
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
                                           cv=StratifiedKFold(y, n_folds=10),
                                           verbose=1,
                                           population_size=50,
                                           gene_mutation_prob=0.10,
                                           gene_crossover_prob=0.5,
                                           tournament_size=3,
                                           generations_number=10)
        cv.fit(X, y)


    Attributes
    ----------
    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.

    best_score_ : float
        Score of best_estimator on the left out data.

    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.
    """
    def __init__(self, estimator, params, scoring=None, cv=4,
                 refit=True, verbose=False, population_size=50,
                 gene_mutation_prob=0.1, gene_crossover_prob=0.5,
                 tournament_size=3, generations_number=10, gene_type=None,
                 n_jobs=1, iid=True, pre_dispatch='2*n_jobs', error_score='raise',
                 fit_params=None):
        super(EvolutionaryAlgorithmSearchCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=fit_params,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score)
        self.params = params
        self.possible_params = params if isinstance(params, list) else [params]
        self.population_size = population_size
        self.generations_number = generations_number
        self._individual_evals = {}
        self.gene_mutation_prob = gene_mutation_prob
        self.gene_crossover_prob = gene_crossover_prob
        self.tournament_size = tournament_size
        self.gene_type = gene_type
        self.score_cache = {}
        self.num_cache_hits = 0
        self.num_evaluations = 0

    def fit(self, X, y=None):
        self.best_estimator_ = None
        self.best_score_ = float("-inf")
        self.best_params_ = None
        for possible_params in self.possible_params:
            self._fit(X, y, possible_params)
        if self.refit:
            self.best_estimator_ = clone(self.estimator)
            self.best_estimator_.set_params(**self.best_params_)
            self.best_estimator_.fit(X, y)

    def _fit(self, X, y, parameter_dict):

        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        n_samples = _num_samples(X)
        X, y = indexable(X, y)

        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))
        cv = check_cv(self.cv, y=y, classifier=is_classifier(self.estimator))

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, est=clone(self.estimator), fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        name_values, gene_type, maxints = _get_param_types_maxint(parameter_dict)
        if self.gene_type is None:
            self.gene_type = gene_type

        if self.verbose:
            print("Types %s and maxint %s detected" % (self.gene_type, maxints))

        toolbox.register("individual", _initIndividual, creator.Individual, maxints=maxints)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", _evalFunction, searchobj=self,
                         name_values=name_values, X=X, y=y,
                         scorer=self.scorer_, cv=cv, iid=self.iid, verbose=self.verbose,
                         error_score=self.error_score, fit_params=self.fit_params)

        toolbox.register("mate", _cxIndividual, indpb=self.gene_crossover_prob, gene_type=self.gene_type)

        toolbox.register("mutate", _mutIndividual, indpb=self.gene_mutation_prob, up=maxints)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        if self.n_jobs > 1:
            pool = Pool(processes=self.n_jobs)
            toolbox.register("map", pool.map)
        pop = toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        if self.verbose:
            print('--- Evolve in {0} possible combinations ---'.format(np.prod(np.array(maxints) + 1)))

        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                           ngen=self.generations_number, stats=stats,
                                           halloffame=hof, verbose=self.verbose)

        current_best_score_ = hof[0].fitness.values[0]
        current_best_params_ = _individual_to_params(hof[0], name_values)

        if self.verbose:
            print("Best individual is: %s\nwith fitness: %s" % (
                current_best_params_, current_best_score_))
            print("Scoring evaluations: %d, Cache hits: %d, Total: %d" % (
                self.num_evaluations, self.num_cache_hits, self.num_evaluations + self.num_cache_hits))

        if current_best_score_ > self.best_score_:
            self.best_score_ = current_best_score_
            self.best_params_ = current_best_params_
