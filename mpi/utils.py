import logging
from mpi4py import MPI 

def generate_logger_MPI(logfile, level):
    logging.basicConfig(filename=logfile, level=level)
    logger = logging.getLogger("rank%i" % MPI.COMM_WORLD.Get_rank())
    return logger
