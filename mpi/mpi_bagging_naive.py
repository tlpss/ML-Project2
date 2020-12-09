    """
    MPI gridsearch for soft bagging. 
    Naive implementation that does not parallellize the ensemble itself, only the different hyperparam evaluations

    run using the .run file on SCITAS or locally using 
    $ mpiexec -n 5 python -m mpi4py.futures mpi/mpi_bagging_naive.py

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

from aggregating.gridsearch import evaluate_model
from aggregating.models import SimpleBagger
from aggregating.utils import normalized_error_VT, flatten_X, generate_V_0, generate_train_set, create_GPR
from mpi.utils import generate_logger_MPI, write_results
from stochastic_models import MaxCallStochasticModel



LOGFILE = "logs/bagging.log"
LOGLEVEL = logging.INFO
WRITEBACK = True
 
class Config:
    """
    container class for config params
    """
    N_train  = 5000
    N_test = 100000
    d = 1
    T = 2
    Delta= [1/12,11/12]
    trials = 2

    #Hyperparam Grid

    M_grid = [1,3,5,7,9,11]
    alpha_grid = [0.2,0.3,0.5,0.7]

def evaluate_model_MPI(*args,**kwargs):
    """
    wrapper around the evaluate_model to allow for creating a logger with the MPI rank

    adds logger to the kwargs and calls the evaluate model function

    :return: evaluate_model(*args,**kwargs)
    :rtype: list(float)
    """
    logger = generate_logger_MPI(LOGFILE,LOGLEVEL)
    kwargs["logger"] = logger
    return evaluate_model(*args,**kwargs)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


logger = generate_logger_MPI(LOGFILE,LOGLEVEL)    
logger.info(f"node with rank {rank} started")   

if rank == 0:
    """
    executed by main MPI process 
    
    mpiexec -n <num_nodes> python -m mpi4py.futures mpi\mpi_bagging.py
    will create 1 dispatcher node with rank 0 and num_node-1 workers for the pool

    """

    ## generate model
    gpr = create_GPR(Config.N_train)
    model = SimpleBagger(1,1,gpr)

    ## generate Training set & V_0
    X_train, y_train = generate_train_set(Config.N_train, Config.Delta,Config.d)
    V_0 = generate_V_0(100000,Config.Delta, Config.d)
    logger.info(f"V_0 = {V_0}")

        

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
                logger.info(f"starting evaluation for {hyperparams}")
                future = executor.submit(evaluate_model_MPI, model,hyperparams,X_train,y_train,Config.d,
                Config.Delta,Config.trials,Config.N_test,MaxCallStochasticModel,V_0 = V_0)
                futures.append([m,alpha,future])
        # get results
        for future in futures:
            results.append([future[0],future[1],future[2].result()])
        
    results.sort(key = lambda x: (x[0],x[1]))
    
    if WRITEBACK:
        write_results("mpi_bagging",results,Config,Config.M_grid,Config.alpha_grid)
    print(results)

