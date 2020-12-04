import logging
from mpi4py import MPI 
import json
import datetime

from aggregating.utils import flatten_X
from stochastic_models import MaxCallStochasticModel

def generate_logger_MPI(logfile, level):
    """
    generate logger for MPI 

    :param logfile: relative path to file
    :type logfile: str
    :param level: logging level (info,debug,..)
    :type level: logging.level
    :return: logger
    :rtype: logging.logger
    """
    logging.basicConfig(filename=logfile, level=level,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(funcName)s : %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',)
    logger = logging.getLogger("rank%i" % MPI.COMM_WORLD.Get_rank())
    return logger

def write_results(basename,results,Config,M_grid,alpha_grid):
    res_dict = {'N_train': Config.N_train, 'N_test': Config.N_test,'mgrid': M_grid, 'alpha_grid': alpha_grid, 'errors': results}
    with open("logs/" + basename + f'{str(datetime.date.today())}.json', 'w') as fp:
        json.dump(res_dict, fp)
