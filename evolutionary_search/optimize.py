# -*- coding: utf-8 -*-
import numpy as np
from deap import base, creator, tools, algorithms
from multiprocessing import Pool
from sklearn.model_selection._search import _check_param_grid

from .cv import _get_param_types_maxint, _initIndividual, _cxIndividual, _mutIndividual, _individual_to_params

__score_cache = {}  # Used for memoization

def _evalFunction(func, individual, name_values, verbose=0, error_score='raise', args={}):
    parameters = _individual_to_params(individual, name_values)
    score = 0

    paramkey = str(individual)
    if paramkey in __score_cache:
        score = __score_cache[paramkey]
    else:
        _parameters = dict(parameters)
        _parameters.update(args)
        if error_score == "raise":
            score = func(**_parameters)
        else:
            try:
                score = func(**_parameters)
            except:
                score = error_score

        __score_cache[paramkey] = score

    return (score,)

def maximize(func, parameter_dict, args={},
            verbose=False, population_size=50,
            gene_mutation_prob=0.1, gene_crossover_prob=0.5,
            tournament_size=3, generations_number=10, gene_type=None,
            n_jobs=1, pre_dispatch='2*n_jobs', error_score='raise'):
    """ Same as _fit in EvolutionarySearchCV but without fitting data. More similar to scipy.optimize."""

    global __score_cache
    _check_param_grid(parameter_dict)
    __score_cache = {}  # Refresh this dict
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    name_values, gene_type, maxints = _get_param_types_maxint(parameter_dict)

    if verbose:
        print("Types %s and maxint %s detected" % (gene_type, maxints))

    toolbox.register("individual", _initIndividual, creator.Individual, maxints=maxints)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", _evalFunction, func,
                     name_values=name_values, verbose=verbose,
                     error_score=error_score, args=args)

    toolbox.register("mate", _cxIndividual, indpb=gene_crossover_prob, gene_type=gene_type)

    toolbox.register("mutate", _mutIndividual, indpb=gene_mutation_prob, up=maxints)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    if n_jobs > 1:
        pool = Pool(processes=n_jobs)
        toolbox.register("map", pool.map)
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    if verbose:
        print('--- Evolve in {0} possible combinations ---'.format(np.prod(np.array(maxints) + 1)))

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                       ngen=generations_number, stats=stats,
                                       halloffame=hof, verbose=verbose)

    current_best_score_ = hof[0].fitness.values[0]
    current_best_params_ = _individual_to_params(hof[0], name_values)

    log = {x: logbook.select(x) for x in logbook.header}  # Convert logbook to pandas compatible dict

    if verbose:
        print("Best individual is: %s\nwith fitness: %s" % (
            current_best_params_, current_best_score_))

    return current_best_params_, current_best_score_, log
