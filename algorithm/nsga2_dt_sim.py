import copy
import sys
import time
from pymoo.core.result import Result
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.problem import Problem
from exception.configuration import RestrictiveConfigException
from visualization import output
from problem.adas_problem import ADASProblem
from problem.pymoo_test_problem import PymooTestProblem
from utils.time_utils import convert_pymoo_time_to_seconds
from pymoo.core.population import Population
from visualization.configuration import *
from algorithm.classification.classifier import ClassificationType
from algorithm.classification.decision_tree import decision_tree
from utils.evaluation import evaluate_individuals
from utils.sorting import *
from utils.time_utils import convert_pymoo_time_to_seconds
from visualization.configuration import *
import numpy as np
import os
from experiment.search_configuration import SearchConfiguration
import quality_indicators.metrics.spread as qi
from model_ga.result import *
import logging as log


'''General flags'''
ALGORITHM_NAME = "NSGA-II-DT"
RESULTS_FOLDER = os.sep + "results" + os.sep
DO_EVAL_APPROXIMATION = True
WRITE_ALL_INDIVIDUALS = False

class NSGAII_DT_SIM(object):

    algorithm_name = ALGORITHM_NAME

    def __init__(self,
                 problem: Problem,
                 config: SearchConfiguration):

        self.problem = problem
        self.config = config
        self.done_tree_iterations = 0
        self.res = None
        
    def run(self) -> ResultExtended:
        problem = self.problem
        config = self.config

        population_size = config.population_size
        maximal_execution_time = config.maximal_execution_time
        max_tree_iterations = config.max_tree_iterations
        num_offsprings = config.num_offsprings
        prob_crossover = config.prob_crossover
        eta_crossover = config.eta_crossover
        prob_mutation = config.prob_mutation
        eta_mutation = config.eta_mutation
        inner_num_gen = config.inner_num_gen

        '''Output variables'''
        all_population = Population()
        best_population = Population()
        best_population_unranked = Population()

        '''Initial conditions (initial region)'''
        xl = problem.xl
        xu = problem.xu

        sampling = LHS()  # Latin Hypercube Sampling
        initial_population = sampling(problem, population_size)
        hist_holder = []

        '''Parameters of the algorithm'''
        if prob_mutation is None:
            prob_mutation = 1 / problem.n_var

        '''Parameter for evaluation'''
        if maximal_execution_time is not None:
            _maximal_execution_time = convert_pymoo_time_to_seconds(
                maximal_execution_time)
            max_tree_iterations = sys.maxsize
            log.info("Search is constrained by maximal execution time")
        elif max_tree_iterations is not None:
            _maximal_execution_time = sys.maxsize
            log.info("Search is constrained by maximal number of tree generations")
        else:
            log.info("Parameters are not correctly set, cannot start search.")
            sys.exit()

        ''' Computation start '''
        start_time = time.time()

        if not self.is_time_left_for_evaluation(start_time,
                                    _maximal_execution_time, 
                                    problem.approx_eval_time,
                                    initial_population):
            raise RestrictiveConfigException()

        evaluate_individuals(initial_population, problem)

        initial_region = decision_tree.Region(xl, xu, initial_population)
        critical_regions = [initial_region]

        # check if search config is not too restrictive (enough time to perform at least one iteration)
        if not self.is_time_left_for_next_iteration(start_time,
                                                   _maximal_execution_time, 
                                                   problem.approx_eval_time,
                                                   critical_regions,
                                                   inner_num_gen):
            raise RestrictiveConfigException()
   
        hist_holder = []

        # inner_algorithm is a template for an algorithm object that is stored for every generation
        inner_algorithm = NSGA2(
            pop_size=None,
            n_offsprings=None,
            sampling=None,
            crossover=SBX(prob=prob_crossover, eta=eta_crossover),
            mutation=PM(prob=prob_mutation, eta=eta_mutation),
            eliminate_duplicates=True)

        tree_iteration = 0 
        n_func_evals = 0
        while n_func_evals < config.n_func_evals_lim:
            # extend the history by one generation
            hist_holder.extend([inner_algorithm] * inner_num_gen)

            log.info(f"running iteration {tree_iteration}")
            for critical_region in critical_regions:

                sub_problem = problem
                if prob_mutation == None:
                    prob_mutation = 1 / problem.n_var

                nd_individuals_region = calc_nondominated_individuals(
                    critical_region.population)
                initial_population = Population(
                    individuals=nd_individuals_region)
                pop_size = len(initial_population)

                algorithm = NSGA2(
                    pop_size=pop_size,
                    n_offsprings=num_offsprings,
                    sampling=initial_population,
                    crossover=SBX(prob=prob_crossover, eta=eta_crossover),
                    mutation=PM(prob=prob_mutation, eta=eta_mutation),
                    eliminate_duplicates=True)

                termination = get_termination("n_gen", inner_num_gen)

                res = minimize(sub_problem,
                               algorithm,
                               termination,
                               seed=1,
                               save_history=True,
                               verbose=True)

                n_func_evals += res.history[-1].evaluator.n_eval

                for i in range(inner_num_gen):
                    pop = Population.merge(
                        hist_holder[tree_iteration * inner_num_gen + i].pop, res.history[i].pop)
                    # copy a template of the inner algorithm, and then modify its population and other properties
                    algo = copy.deepcopy(inner_algorithm)
                    algo.pop = pop
                    # algo._set_optimum()
                    opt_pop = Population(
                        individuals=calc_nondominated_individuals(pop))
                    algo.opt = opt_pop
                    # algo.evaluator.n_eval += i + inner_num_gen * tree_iteration
                    hist_holder[tree_iteration * inner_num_gen + i] = algo

                hist = res.history

                # hist[i] is an object of <class 'pymoo.algorithms.moo.nsga2.NSGA2'>
                best_population_unranked = Population.merge(
                    best_population, res.opt)
                best_population = get_nondominated_population(
                    best_population_unranked)
                for generation in hist:
                    all_population = Population.merge(
                        all_population, generation.pop)

            initial_region.population = best_population

            regions = decision_tree.generate_critical_regions(
                all_population, problem, save_folder=None)
            critical_regions = [
                region for region in regions if region.is_critical]
            if not critical_regions:
                critical_regions = [initial_region]
            tree_iteration += 1

        execution_time = time.time() - start_time

        # store the number of achieved tree iterations
        self.done_tree_iterations = tree_iteration

        I = 0
        for algo in hist_holder:
            I += len(algo.pop)
            algo.evaluator.n_eval = I
            algo.start_time = 0
            algo.problem = problem
            algo.result()

        res_holder = ResultExtended()
        res_holder.algorithm = inner_algorithm
        res_holder.algorithm.evaluator.n_eval = I
        res_holder.problem = problem
        res_holder.algorithm.problem = problem
        res_holder.history = hist_holder
        res_holder.exec_time = execution_time

        # calculate total optimal population using individuals from all iterations
        opt_all = Population()
        for algo in hist_holder:
            opt_all = Population.merge(opt_all, algo.pop)
        opt_all_nds = get_nondominated_population(opt_all)
        res_holder.opt = opt_all_nds
        self.res = res_holder

        return res_holder

    def write_results(self, results_folder=RESULTS_FOLDER):
        algorithm_name = self.algorithm_name
        if self.res is None:
            log.info(
                "Result object is None. Execute algorithm first, before writing results.")
            return

        res_holder = self.res
        config = self.config
        problem = self.problem
        hist_holder = res_holder.history

        done_tree_iterations = self.done_tree_iterations

        '''For forwarding to plotter'''
        algorithm_parameters = {
            'Number of maximal tree generations': str(config.max_tree_iterations),
            'Number performed tree iterations': done_tree_iterations,
            "Population size": str(config.population_size),
            "Number of generations": str(config.inner_num_gen),
            "Number of offsprings": str(config.num_offsprings),
            "Crossover probability": str(config.prob_crossover),
            "Crossover eta": str(config.eta_crossover),
            "Mutation probability": str(config.prob_mutation),
            "Mutation eta": str(config.eta_mutation)}

        log.info(f"=====[{algorithm_name}] Writing results...")

        save_folder = output.create_save_folder(
            problem, results_folder, algorithm_name)

        # output of plots for every tree iteration
        for i in range(done_tree_iterations):
            res_holder_tmp = copy.deepcopy(res_holder)
            res_holder_tmp.history = hist_holder[0:(
                i + 1) * config.inner_num_gen]

            opt_all_tmp = Population()
            for algo in res_holder_tmp.history:
                opt_all_tmp = Population.merge(opt_all_tmp, algo.pop)
            opt_all_tmp_nds = get_nondominated_population(opt_all_tmp)
            res_holder_tmp.opt = opt_all_tmp_nds

            output.design_space(res_holder_tmp, save_folder +
                                "tree_iterations" + os.sep + "TI" + str(i) + os.sep)
            output.objective_space(
                res_holder_tmp, save_folder + "tree_iterations" + os.sep + "TI" + str(i) + os.sep)

        output.write_summary_results(res_holder, save_folder)
        output.igd_analysis(res_holder, save_folder)
        output.hypervolume_analysis(res_holder, save_folder)
        output.spread_analysis(res_holder, save_folder)
        output.write_calculation_properties(
            res_holder, save_folder, algorithm_name, algorithm_parameters)
        output.design_space(res_holder, save_folder)
        output.objective_space(res_holder, save_folder)
        output.optimal_individuals(res_holder, save_folder)
        output.all_critical_individuals(res_holder, save_folder)

        if WRITE_ALL_INDIVIDUALS:
            output.all_individuals(res_holder, save_folder)

        #persist results object
        res_holder.persist(save_folder + "backup")

    ''' Approximate whether time has left for the next tree iteration by using simulation time '''
    def is_time_left_for_next_iteration(self, start_time, _maximal_execution_time, simulation_time, critical_regions,
                                        inner_num_gen):
        if DO_EVAL_APPROXIMATION:
            tolerance = 1  # in seconds
            time_min = _maximal_execution_time - (time.time() - start_time)
            time_real = np.sum([len(region.population)
                                for region in critical_regions]) * simulation_time * inner_num_gen
            log.info(f"Time left for next iteration: {time_min}")
            log.info(f"Time required for next iteration: {time_real}")
            return time_real < time_min + tolerance
        else:
            return (time.time() - start_time) < _maximal_execution_time

    ''' Approximate whether time has left for the next tree iteration by using simulation time '''
    def is_time_left_for_evaluation(self, start_time, _maximal_execution_time, simulation_time, population):
        if DO_EVAL_APPROXIMATION:
            tolerance = 1  # in seconds
            time_min = _maximal_execution_time - (time.time() - start_time)
            time_real = len(population)*simulation_time
            log.info(f"Time left for evaluation of population: {time_min}")
            log.info(f"Time required for evaluation of population: {time_real}")
            return time_real < time_min + tolerance
        else:
            return (time.time() - start_time) < _maximal_execution_time
