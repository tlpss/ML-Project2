import logging
from mpi4py import MPI 

def generate_logger_MPI(logfile, level):
    logging.basicConfig(filename=logfile, level=level,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(funcName)s : %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',)
    logger = logging.getLogger("rank%i" % MPI.COMM_WORLD.Get_rank())
    return logger
