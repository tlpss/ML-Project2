'''
dummpy MPI file to showcase how to run an MPIPool, keeping gridsearches in mind
a
the goal is to show 
1) how to pass arguments
2) how to get the results back 
3) what it means to have different processes, illustrated by the global counter

how to run this file?

https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html

$ mpiexec -n 1 -usize 5 python mpi/mpi_dummy.py 
if your mpi distro allows the -usize keyword

$ mpiexec -n 5 python -m mpi4py.futures mpi/mpi_dummy.py
if not (eg MSMPI)

for each of the N processes
''' 

from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
import logging 

# add modules to path
import os 
import sys         
module_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(module_path)

from aggregating.gridsearch import evaluate_model
from mpi.utils import generate_logger_MPI


counter = 0 
LOGFILE = "logs/dummy.log"
LOGLEVEL = logging.INFO


def dummy(input,i):
    global counter 
    counter +=1
    logger = generate_logger_MPI(LOGFILE,LOGLEVEL,rank)                                
    logger.info(f"input={input},i= {i}, rank={MPI.COMM_WORLD.Get_rank()}")
    logger.info(f"counter value = {counter}")
    return [counter]*2

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


logger = generate_logger_MPI(LOGFILE,LOGLEVEL,rank)    
logger.info(f"node with rank {rank} started")                          

hyperparams = [{'M':1},{'M':3},{'M':5},{'M':7},{'M':9},{'M':11},{'M':13}]

if rank == 0:
    with MPIPoolExecutor() as executor:
        futures =  []
        for index,hyperparam in enumerate(hyperparams):
            logger.info(f"starting executor with {hyperparam}")
            futures.append(executor.submit(dummy, hyperparam,3))
        results  = []
        for future in futures:
            results.append(future.result())

        print(results)


