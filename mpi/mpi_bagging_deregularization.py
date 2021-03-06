"""
Parallelized MPI version to evaluate the effect of fixing Lambda
Here we perform a gridsearch over different lambda ranges in order to see if the ensemble can leverage its reduced variance to allow a higher variance
of a single estimator (which implies lower bias)
"""

# add modules to path
import os 
import sys         
module_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(module_path)

import logging 
import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor

from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI

from aggregating.gridsearch import evaluate_model
from aggregating.models import SimpleBagger
from aggregating.utils import create_reg_kernel, normalized_error_VT, flatten_X, generate_V_0, generate_train_set, create_GPR
from mpi.utils import generate_logger_MPI, trials_hard_prediction, write_results, generate_bagging_train_indices,train_and_evaluate, generate_test_sets, trials_soft_prediction
from stochastic_models import MaxCallStochasticModel

np.random.seed(2020)

LOGFILE = "logs/degeg_bagging.log"
LOGLEVEL = logging.INFO
WRITEBACK = True
SOFTVOTING = True
 
class Config:
    """
    container class for config params
    """
    N_train = 20000
    N_test = 100000
    d = 6
    T = 2
    Delta= [1/12,11/12]
    trials = 2

    #Hyperparam Grid

    M_grid = [1,4,7,10,13,16,19]
    alpha_grid = [0.5] # legacy for the results writer
    alpha = 0.5
    lambda_grid = [1e-12,1e-10, 1e-8, 1e-6,1e-4]


class DataContainer:
    """
    holds data that is shared between the MPI tasks
    """
    X_train  = np.empty((Config.N_train,Config.d*Config.T)) 
    y_train = np.empty((Config.N_train,1))
    X_test_list = np.empty((Config.trials,Config.N_test,Config.d*Config.T))
    y_test_list = np.empty((Config.trials,Config.N_test,1))



def train_and_evaluate_wrapper(model, train_indices,alpha,m,i):
    """
    wrapper around the evaluate_model to allow for creating a logger with the MPI rank

    adds logger to the kwargs and calls the evaluate model function

    :return: evaluate_model(*args,**kwargs)
    :rtype: list(float)
    """
    logger = generate_logger_MPI(LOGFILE,LOGLEVEL,rank)
    logger.info(f"executing{i}-th train_evaluate for M,alpha= {m},{alpha} ")
    X_train, y_train = DataContainer.X_train[train_indices],DataContainer.y_train[train_indices]
    return train_and_evaluate(model, X_train, y_train, DataContainer.X_test_list)

## init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logger = generate_logger_MPI(LOGFILE,LOGLEVEL,rank)    
logger.info(f"node with rank {rank}/{size} started")  

## let the main task create the train & testsets
if rank == 0:
    logger.info(f"creating train & testsets")
    DataContainer.X_train, DataContainer.y_train = generate_train_set(Config.N_train, Config.Delta,Config.d)
    DataContainer.X_test_list, DataContainer.y_test_list = generate_test_sets(Config.trials, Config.N_test,Config.Delta, Config.d)


## broadcast the required data to all nodes
# broadcast the numpy arrays separately for efficiency gains
# make broadcasts non blocking since the worker nodes are spawned at different times
# https://github.com/mpi4py/mpi4py/blob/70333ef76db05f643347b9880a05967891fb1eed/src/mpi4py/MPI/Comm.pyx#L750 
# this feature is not documented in the documentation, but the source code clearly indicates it is present
xtrain_req = comm.Ibcast(DataContainer.X_train, root = 0)
ytrain_req = comm.Ibcast(DataContainer.y_train, root = 0)
ytest_req = comm.Ibcast(DataContainer.X_test_list,root = 0)

if rank > 0:
    # want the broadcast to be blocking since we need the data before continuing
    logger.debug(f"waiting for broadcasts")
    xtrain_req.Wait()
    ytrain_req.Wait()
    ytest_req.Wait()


logger.debug(f"X_test_list= {DataContainer.X_test_list}")
logger.info(f"broadcasting done for this node")

if rank == 0:
    """
    executed by main MPI process 
    
    mpiexec -n <num_nodes> python -m mpi4py.futures mpi\mpi_bagging.py
    will create 1 dispatcher node with rank 0 and num_node-1 workers for the pool

    """

    ## create logger
    logger = generate_logger_MPI(LOGFILE,LOGLEVEL,rank)                                

    ## MPI execute
    results = []
    with MPIPoolExecutor() as executor:
        futures =  []
        # evaluate model for all points in grid by creating new mpi node
        for m in Config.M_grid:
            for lambda_ in Config.lambda_grid:
                hyperparams= {'M':m, 'train_size_alpha':Config.alpha, "lambda": lambda_}
                logger.info(f"starting evaluations for {hyperparams}")
                # generate index sets for the ensemble
                indices_list  = generate_bagging_train_indices(Config.N_train,Config.alpha,m)
                predictor_futures = []

                for i in range(m):
                    # clone base model for each member of the ensemble
                    reg_kernel = create_reg_kernel(lambda_)
                    gpr = GaussianProcessRegressor(reg_kernel,  copy_X_train=False)
                    # submit job 
                    logger.debug(f"starting {i}-th model of the ensemble")
                    future = executor.submit(train_and_evaluate_wrapper, gpr,indices_list[i],Config.alpha, m, i) #[mu_list,sigma_list] each of len trials
                    predictor_futures.append(future) 

                futures.append([m,lambda_,predictor_futures]) # M* [mu_list,sigma_list] each of len trials
        # get results
        for future in futures:
            predictor_results = []
            for prediction in future[2]:
                prediction_result = prediction.result()
                predictor_results.append(prediction_result)
            results.append([future[0],future[1],predictor_results])
    results.sort(key = lambda x: (x[0],x[1]))

    # combine the ensemble predictions into a single prediction for each trial of each hyperparam setting
    #  
    predictions = []
    for result in results:
        predictor_results = result[2] #M* [(m_ij,s_ij)] each of len trials
        if SOFTVOTING:
            bagging_predictions = trials_soft_prediction(predictor_results, Config.trials)
        else: 
            bagging_predictions = trials_hard_prediction(predictor_results,Config.trials)
    
        predictions.append([result[0],result[1],bagging_predictions])

    ## compute the normalized error for all hyperparam runs & all trials
    V_0  = generate_V_0(100000,Config.Delta,Config.d)
    logger.info(f"V_0 = {V_0}")
    normalized_error_results = []
    for result in predictions:
        errors  = []
        for index,prediction in enumerate(result[2]):
            error = normalized_error_VT(prediction,DataContainer.y_test_list[index],V_0)
            errors.append(error)
        normalized_error_results.append([result[0],result[1],errors])

    print(normalized_error_results)
    if WRITEBACK:
        write_results("mpi_regularized_bagging",normalized_error_results,Config)

