
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
import logging 

# add modules to path
import os 
import sys         
module_path = os.path.abspath(os.path.join('...'))
sys.path.append(module_path)

from aggregating.gridsearch import evaluate_model
from mpi.utils import generate_logger_MPI


counter = 0 
LOGFILE = "logs/dummy.log"
LOGLEVEL = logging.INFO

'''
dummpy MPI file to showcase how to run an MPIPool, keeping gridsearches in mind

the goal is to show 
1) how to pass arguments
2) how to get the results back 
3) what it means to have different processes, illustrated by the global counter

how to run this file?

https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html

$ mpiexec -n 1 -usize 5 python mpi/dummy.py 
if your mpi distro allows the -usize keyword

$ mpiexec -n 5 python -m mpi4py.futures mpi/dummy.py
if not (eg MSMPI)

for each of the N processes
''' 

def dummy(input):
    global counter 
    counter +=1
    logger = generate_logger_MPI(LOGFILE,LOGLEVEL)                                
    logger.info(f"input={input},counter= {counter}, rank={MPI.COMM_WORLD.Get_rank()}")
    logger.info(f"counter value = {counter}")
    return [counter]*2

if __name__ == "__main__":
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    logger = generate_logger_MPI(LOGFILE,LOGLEVEL)                                

    hyperparams = [{'M':1},{'M':3},{'M':5},{'M':7},{'M':9},{'M':11},{'M':13}]
    with MPIPoolExecutor() as executor:
        futures =  []
        for index,hyperparam in enumerate(hyperparams):
            logger.info(f"starting executor with {hyperparam}")
            futures.append(executor.submit(dummy, hyperparam))
        results  = []
        for future in futures:
            results.append(future.result())
    
    print(results)


