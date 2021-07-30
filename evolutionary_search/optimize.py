# -*- coding: utf-8 -*-
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.model_selection._search import _check_param_grid
from .cv import (
    _get_param_types_maxint,
    _initIndividual,
    _cxIndividual,
    _mutIndividual,
    _individual_to_params,
)
import warnings
import os


def _evalFunction(
    func, individual, name_values, verbose=0, error_score="raise", args={}
):
    parameters = _individual_to_params(individual, name_values)
    score = 0

    _parameters = dict(parameters)
    _parameters.update(args)
    if error_score == "raise":
        score = func(**_parameters)
    else:
        try:
            score = func(**_parameters)
        except Exception:
            score = error_score

    return (score,)


def compile():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)


def maximize(
    func,
    parameter_dict,
    args={},
    verbose=False,
    population_size=50,
    gene_mutation_prob=0.1,
    gene_crossover_prob=0.5,
    tournament_size=3,
    generations_number=10,
    gene_type=None,
    n_jobs=1,
    error_score="raise",
):
    """Same as _fit in EvolutionarySearchCV but without fitting data. More similar to scipy.optimize.

    Parameters
    ------------------
    n_jobs : int or map function, default=1
        Number of jobs to run in parallel.
        Also accepts custom parallel map functions from Pool or SCOOP.

    Returns
    ------------------
    best_params_ : dict
        A list of parameters for the best learner.

    best_score_ : float
        The score of the learner described by best_params_

    score_results : tuple of 2-tuples ((dict, float), ...)
        The score of every individual evaluation indexed by it's parameters.

    hist : deap.tools.History object.
        Use to get the geneology data of the search.

    logbook: deap.tools.Logbook object.
        Includes the statistics of the evolution.
    """

    toolbox = base.Toolbox()

    _check_param_grid(parameter_dict)
    if isinstance(n_jobs, int):
        # If n_jobs is an int, greater than 1 or less than 0 (indicating to use as
        # many jobs as possible) then we are going to create a default pool.
        # Windows users need to be warned of this feature as it only works properly
        # on linux. They need to encapsulate their pool in an if __name__ == "__main__"
        # wrapper so that pools are not recursively created when the module is reloaded in each map
        if isinstance(n_jobs, (int, float)):
            if n_jobs > 1 or n_jobs < 0:
                from multiprocessing import Pool  # Only imports if needed

                if os.name == "nt":  # Checks if we are on Windows
                    warnings.warn(
                        (
                            "Windows requires Pools to be declared from within "
                            "an 'if __name__==\"__main__\":' structure. In this "
                            "case, n_jobs will accept map functions as well to "
                            "facilitate custom parallelism. Please check to see "
                            "that all code is working as expected."
                        )
                    )
                pool = Pool(n_jobs)
                toolbox.register("map", pool.map)
                warnings.warn("Need to create a creator. Run optimize.compile()")
            else:
                compile()

    # If it's not an int, we are going to pass it as the map directly
    else:
        try:
            toolbox.register("map", n_jobs)
        except Exception:
            raise TypeError(
                "n_jobs must be either an integer or map function. Received: {}".format(
                    type(n_jobs)
                )
            )

    name_values, gene_type, maxints = _get_param_types_maxint(parameter_dict)

    if verbose:
        print("Types %s and maxint %s detected" % (gene_type, maxints))

    toolbox.register("individual", _initIndividual, creator.Individual, maxints=maxints)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        _evalFunction,
        func,
        name_values=name_values,
        verbose=verbose,
        error_score=error_score,
        args=args,
    )

    toolbox.register(
        "mate", _cxIndividual, indpb=gene_crossover_prob, gene_type=gene_type
    )

    toolbox.register("mutate", _mutIndividual, indpb=gene_mutation_prob, up=maxints)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    # Tools
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)
    stats.register("std", np.nanstd)

    # History
    hist = tools.History()
    toolbox.decorate("mate", hist.decorator)
    toolbox.decorate("mutate", hist.decorator)
    hist.update(pop)

    if verbose:
        print(
            "--- Evolve in {0} possible combinations ---".format(
                np.prod(np.array(maxints) + 1)
            )
        )

    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=generations_number,
        stats=stats,
        halloffame=hof,
        verbose=verbose,
    )

    current_best_score_ = hof[0].fitness.values[0]
    current_best_params_ = _individual_to_params(hof[0], name_values)

    # Generate score_cache with real parameters
    _, individuals, each_scores = zip(
        *[
            (idx, indiv, np.mean(indiv.fitness.values))
            for idx, indiv in list(hist.genealogy_history.items())
            if indiv.fitness.valid and not np.all(np.isnan(indiv.fitness.values))
        ]
    )
    unique_individuals = {
        str(indiv): (indiv, score) for indiv, score in zip(individuals, each_scores)
    }
    score_results = tuple(
        [
            (_individual_to_params(indiv, name_values), score)
            for indiv, score in unique_individuals.values()
        ]
    )

    if verbose:
        print(
            "Best individual is: %s\nwith fitness: %s"
            % (current_best_params_, current_best_score_)
        )

    # Close your pools if you made them
    if isinstance(n_jobs, int) and (n_jobs > 1 or n_jobs < 0):
        pool.close()
        pool.join()

    return current_best_params_, current_best_score_, score_results, hist, logbook
