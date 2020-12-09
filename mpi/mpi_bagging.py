"""
Parallelized MPI gridsearch for SOFT BAGGING
"""

# add modules to path
import os 
import sys         
module_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(module_path)

import logging 
import numpy as np 
from sklearn.base import clone

from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI

from aggregating.gridsearch import evaluate_model
from aggregating.models import SimpleBagger
from aggregating.utils import normalized_error_VT, flatten_X, generate_V_0, generate_train_set, create_GPR
from mpi.utils import generate_logger_MPI, write_results, generate_bagging_train_indices,train_and_evaluate, generate_test_sets, trials_soft_prediction
from stochastic_models import MaxCallStochasticModel



LOGFILE = "logs/bagging.log"
LOGLEVEL = logging.INFO
WRITEBACK = True
 
class Config:
    """
    container class for config params
    """
    N_train  = 500
    N_test = 5000
    d = 1
    T = 2
    Delta= [1/12,11/12]
    trials = 2

    #Hyperparam Grid

    M_grid = [1,3,5,7,9]
    alpha_grid = [0.3]

class DataContainer:
    """
    holds data that is shared between the MPI tasks
    """
    X_train  = np.empty((Config.N_train,Config.d*Config.T)) 
    y_train = np.empty((Config.N_train,1))
    X_test_list = [np.empty((Config.N_test,Config.d*Config.T))] * Config.trials
    y_test_list = [np.empty((Config.N_test,1))] * Config.trials



def train_and_evaluate_wrapper(model, train_indices,alpha,m,i):
    """
    wrapper around the evaluate_model to allow for creating a logger with the MPI rank

    adds logger to the kwargs and calls the evaluate model function

    :return: evaluate_model(*args,**kwargs)
    :rtype: list(float)
    """
    logger = generate_logger_MPI(LOGFILE,LOGLEVEL)
    logger.info(f"executing{i}-th train_evaluate for M,alpha= {m},{alpha} ")

    X_train, y_train = DataContainer.X_train[train_indices],DataContainer.y_train[train_indices]
    return train_and_evaluate(model, X_train, y_train, DataContainer.X_test_list)

## init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logger = generate_logger_MPI(LOGFILE,LOGLEVEL)    
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

xtest_list_req = []
for i in range(len(DataContainer.X_test_list)):
    req = comm.Ibcast(DataContainer.X_test_list[i],root = 0)
    xtest_list_req.append(req)

if rank > 0:
    # want the broadcast to be blocking since we need the data before continuing
    logger.debug(f"waiting for broadcasts")
    xtrain_req.Wait()
    ytrain_req.Wait()
    for req in xtest_list_req:
        req.Wait()


logger.debug(f"X_test_list= {DataContainer.X_test_list}")
logger.info(f"broadcasting done for this node")

if rank == 0:
    """
    executed by main MPI process 
    
    mpiexec -n <num_nodes> python -m mpi4py.futures mpi\mpi_bagging.py
    will create 1 dispatcher node with rank 0 and num_node-1 workers for the pool

    """

    ## generate model
    base_gpr = create_GPR(Config.N_train)

    ## create logger
    logger = generate_logger_MPI(LOGFILE,LOGLEVEL)                                

    ## MPI execute
    results = []
    with MPIPoolExecutor() as executor:
        futures =  []
        # evaluate model for all points in grid by creating new mpi node
        for m in Config.M_grid:
            for alpha in Config.alpha_grid:
                hyperparams= {'M':m, 'train_size_alpha':alpha}
                logger.info(f"starting evaluations for {hyperparams}")
                # generate index sets for the ensemble
                indices_list  = generate_bagging_train_indices(Config.N_train,alpha,m)
                predictor_futures = []

                for i in range(m):
                    # clone base model for each member of the ensemble
                    gpr = clone(base_gpr)
                    # submit job 
                    logger.debug(f"starting {i}-th model of the ensemble")
                    future = executor.submit(train_and_evaluate_wrapper, gpr,indices_list[i],alpha, m, i) #[mu_list,sigma_list] each of len trials
                    predictor_futures.append(future) 

                futures.append([m,alpha,predictor_futures]) # M* [mu_list,sigma_list] each of len trials
        # get results
        for future in futures:
            predictor_results = []
            for prediction in future[2]:
                prediction_result = prediction.result()
                predictor_results.append(prediction_result)
            results.append([future[0],future[1],predictor_results])
    results.sort(key = lambda x: (x[0],x[1]))


    # combine the ensemble predictions into a single prediction for each trial of each hyperparam setting
    # SOFT BAGGING
    predictions = []
    for result in results:
        predictor_results = result[2] #M* [mu_list,sigma_list] each of len trials
        bagging_predictions = trials_soft_prediction(predictor_results, Config.trials)
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
        write_results("mpi_bagging",normalized_error_results,Config)

