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
LOGLEVEL = logging.DEBUG
WRITEBACK = False
 
class Config:
    """
    container class for config params
    """
    N_train  = 20
    N_test = 5
    d = 3
    T = 2
    Delta= [1/12,11/12]
    trials = 2

    #Hyperparam Grid

    M_grid = [2,3]
    alpha_grid = [0.2]

class DataContainer:
    """
    holds data that is shared between the MPI tasks
    """
    X_train  = np.empty((Config.N_train,Config.d*Config.T)) 
    y_train = np.empty((Config.N_train,1))
    X_test_list = [np.empty((Config.N_test,Config.d*Config.T))] * Config.trials
    y_test_list = [np.empty((Config.N_test,1))] * Config.trials



def train_and_evaluate_wrapper(model, train_indices):
    """
    wrapper around the evaluate_model to allow for creating a logger with the MPI rank

    adds logger to the kwargs and calls the evaluate model function

    :return: evaluate_model(*args,**kwargs)
    :rtype: list(float)
    """
    logger = generate_logger_MPI(LOGFILE,LOGLEVEL)
    logger.debug("executing train_evaluate")

    X_train, y_train = DataContainer.X_train[train_indices],DataContainer.y_train[train_indices]
    return train_and_evaluate(model, X_train, y_train, DataContainer.X_test_list)

## init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

logger = generate_logger_MPI(LOGFILE,LOGLEVEL)    
logger.info(f"node with rank {rank} started")  

## let the main task create the train & testsets
if rank == 0:
    logger.debug(f"creating train & testsets")
    DataContainer.X_train, DataContainer.y_train = generate_train_set(Config.N_train, Config.Delta,Config.d)
    DataContainer.X_test_list, DataContainer.y_test_list = generate_test_sets(Config.trials, Config.N_test,Config.Delta, Config.d)


## broadcast the required data to all nodes
# broadcast the numpy arrays separately for efficiency gains
comm.Bcast(DataContainer.X_train, root = 0)
comm.Bcast(DataContainer.y_train, root = 0)
for i in range(len(DataContainer.X_test_list)):
    comm.Bcast(DataContainer.X_test_list[i],root = 0)
logger.debug(f"X_test_list= {DataContainer.X_test_list}")
 

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
                # generate index sets
                indices_list  = generate_bagging_train_indices(Config.N_train,alpha,m)
                predictor_futures = []

                for i in range(m):
                    # clone base model 
                    gpr = clone(base_gpr)
                    # submit job
                    logger.debug(f"starting {i}-th model of the ensemble")
                    future = executor.submit(train_and_evaluate_wrapper, gpr,indices_list[i]) #[mu_list,sigma_list] each of len trials
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


    #convert into a single prediction
    predictions = []
    for result in results:
        predictor_results = result[2] #M* [mu_list,sigma_list] each of len trials
        bagging_predictions = trials_soft_prediction(predictor_results, Config.trials)
        predictions.append([result[0],result[1],bagging_predictions])

    ## compute the actual error
    print(f"final predictions = {predictions}")

    if WRITEBACK:
        write_results("mpi_bagging",results,Config)

