"""
MPI boosting

run using the .run file on SCITAS or locally using 
$ mpiexec -n 2 python -m mpi4py.futures mpi/mpi_boosting.py

"""
# add modules to path
import os 
import sys         
module_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(module_path)

import logging 
import numpy as np 

from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI

from boosting.boosting_model_evaluation import *
from aggregating.utils import normalized_error_VT, flatten_X, generate_V_0, generate_train_set, create_GPR
from mpi.utils import generate_logger_MPI, write_boosting_results
from stochastic_models import MaxCallStochasticModel



LOGFILE = "logs/boosting_second_version.log"
LOGLEVEL = logging.INFO
WRITEBACK = True
 
class Config:
    """
    container class for config params
    """
    N_train  = 20000
    N_test = 100000
    d = 6
    T = 2
    Delta= [1/12,11/12]
    Max_Iter = 40
    Epsilon = 1e-8
    Early_Stop = 8
    #Hyperparam 
    Ratios = np.linspace(0.45, 0.9 , num=6).tolist()
    learning_rate = 0.366
    
def evaluate_model_MPI(*args,**kwargs):
    """
    wrapper around the evaluate_model to allow for creating a logger with the MPI rank

    adds logger to the kwargs and calls the evaluate model function

    :return: evaluate_model(*args,**kwargs)
    :rtype: list(float)
    """
    logger = generate_logger_MPI(LOGFILE,LOGLEVEL,rank)
    kwargs["logger"] = logger
    return evaluate_boosting(*args,**kwargs)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

 ## create logger

logger = generate_logger_MPI(LOGFILE,LOGLEVEL,rank)    
logger.info(f"node with rank {rank} started")   

if rank == 0:
    """
    executed by main MPI process 
    
    mpiexec -n <num_nodes> python -m mpi4py.futures mpi\mpi_bagging.py
    will create 1 dispatcher node with rank 0 and num_node-1 workers for the pool

    """

    ## generate Training set, Test set & V_0s
    X_train, y_train = generate_train_set(Config.N_train, Config.Delta,Config.d)
    X_test, y_test = generate_test_set(Config.N_test, Config.Delta,Config.d)

    V_0_train = generate_V_0(Config.N_train,Config.Delta, Config.d)
    V_0_test = generate_V_0(Config.N_test,Config.Delta, Config.d)

    logger.info(f"V_0_test = {V_0_test}")
    
    reference = create_GPR(Config.N_train)
    reference.fit(X_train, y_train)
    f_X = reference.predict(X_test)
    reference_error = normalized_error_VT(f_X, y_test, V_0_test)
    logger.info(f"reference error : {reference_error}")
    
    ## MPI execute
    results = []
    
    with MPIPoolExecutor() as executor:
        futures =  []
        # evaluate model for all points in grid by creating new mpi node
        for r in Config.Ratios:
            logger.info(f"starting evaluation for ratio {r}")
            future = executor.submit(evaluate_boosting, X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy(),                         V_0_train.copy(), Config.Max_Iter,(float("inf"), float("inf")), Config.Early_Stop, Config.learning_rate,                           Config.Epsilon, round(Config.N_train*r),V_0_test= V_0_test.copy(), logger = logger)
            logger.info(f"minimum error found for ratio {r} is {future.result()[2][0]} ")
            futures.append([r,future])
        
        # get results
        for future in futures:
            results.append([future[0],future[1].result()[1],future[1].result()[2]])
        
    results.sort(key = lambda x: x[0])
    
    if WRITEBACK:
        write_boosting_results("mpi_boosting",results,Config)
    print(results)

        