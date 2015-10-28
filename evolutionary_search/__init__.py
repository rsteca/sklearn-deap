# -*- coding: utf-8 -*-
from __future__ import (
    print_function, unicode_literals, division, absolute_import)

from bitstring import BitArray
from deap import base, creator, tools, algorithms
from math import log, ceil
from multiprocessing import Pool
from sklearn import cross_validation
from sklearn.base import clone
from sklearn.grid_search import ParameterGrid, _check_param_grid, BaseSearchCV
import copy_reg
import numpy as np
import random
import types

def _reduce_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

class EvolutionaryAlgorithmSearchCV(BaseSearchCV):

    def __init__(self, estimator, param_grid, scoring=None, cv=4,
                 refit=True, verbose=False, population_size=50, mutation_prob=0.10,
                 tournament_size=3, generations_number=10,
                 n_jobs=1, iid=True, pre_dispatch='2*n_jobs', error_score='raise',
                 fit_params=None):
        super(EvolutionaryAlgorithmSearchCV, self).__init__(
            estimator, scoring, fit_params, n_jobs, iid,
            refit, cv, pre_dispatch, error_score)
        _check_param_grid(param_grid)
        self.param_grid = param_grid
        self.possible_params = list(ParameterGrid(self.param_grid))
        self.individual_size = int(ceil(log(len(self.possible_params), 2)))
        self.population_size = population_size
        self.generations_number = generations_number
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None
        self._individual_evals = {}
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size

    def _individual_to_params(self, val):
        if val >= len(self.possible_params):
            return False
        return self.possible_params[val]

    def _evalFunction(self, individual):
        individual_int = BitArray(individual).uint
        if individual_int not in self._individual_evals:
            params = self._individual_to_params(individual_int)
            if not params:
                return (0,)
            clf = clone(self.estimator)
            clf.set_params(**params)
            scores = cross_validation.cross_val_score(clf, self.X, self.y,
                                                      cv=self.cv, scoring=self.scoring)
            self._individual_evals[individual_int] = float(scores.mean())

        return (self._individual_evals[individual_int],)

    def fit(self, X, y=None):
        self.X = X
        self.y = y

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                      toolbox.attr_bool, n=self.individual_size)
        toolbox.register("population", tools.initRepeat, list,
                      toolbox.individual)

        toolbox.register("evaluate", self._evalFunction)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.mutation_prob)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        if self.n_jobs > 1:
            copy_reg.pickle(types.MethodType, _reduce_method)
            pool = Pool(processes=self.n_jobs)
            # self.toolbox.register("map", parmap)
            toolbox.register("map", pool.map)
        pop = toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        if self.verbose:
            print('--- Evolve in {0} possible combinations ---'.format(len(self.possible_params)))

        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                           ngen=self.generations_number, stats=stats,
                                           halloffame=hof, verbose=self.verbose)

        self.best_score_ = hof[0].fitness
        self.best_params_ = self._individual_to_params(BitArray(hof[0]).uint)

        if self.verbose:
            import json
            print("Best individual is: %s\nwith fitness: %s" % (
                  json.dumps(self.best_params_), hof[0].fitness)
            )

        if self.refit:
            self.best_estimator_ = clone(self.estimator)
            self.best_estimator_.set_params(**self.best_params_)
            self.best_estimator_.fit(self.X, self.y)

    def _fit(self, X, y, parameter_iterable):
        raise NotImplementedError
