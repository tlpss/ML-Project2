import logging
import json
import datetime
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor

from aggregating.utils import flatten_X, generate_train_set, memory_efficient_predict
from stochastic_models import MaxCallStochasticModel

### general MPI helpers
def generate_logger_MPI(logfile, level,rank):
    """
    generate logger for MPI 

    :param logfile: relative path to file
    :type logfile: str
    :param level: logging level (info,debug,..)
    :type level: logging.level
    :param rank: the rank of the process for which to create a logger
    :return: logger
    :rtype: logging.logger
    """
    logging.basicConfig(filename=logfile, level=level,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(funcName)s : %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',)
    logger = logging.getLogger("rank%i" % rank )
    return logger

def write_results(basename,results,Config):
    res_dict = {'N_train': Config.N_train, 'N_test': Config.N_test,'mgrid': Config.M_grid, 'alpha_grid': Config.alpha_grid, 'errors': results}
    with open("logs/" + basename + f'{str(datetime.datetime.now().strftime("%Y-%m-%d.%H-%M-%S"))}.json', 'w') as fp:
        json.dump(res_dict, fp)

#### Bagging MPI helpers
def generate_bagging_train_indices(N_train,alpha,M):
    """
    creates indices that represent M trainingsets, used for bagging (replacement within and in between the sets)

    :param N: trainset len
    :type X: int
    :param alpha: fractional size of each trainset
    :type alpha: float 
    :param M: number of trainsets to create
    :type M: int
    :return: list of indices that represent the M trainsets
    :rtype: List[np.ndarray(M*alpha)]
    """
    n = round(N_train*alpha)
    indices_list  = []
    for i in range(M):
        indices = np.random.choice(N_train,size=n,replace=True)
        indices_list.append(indices)
    return indices_list

def generate_test_sets(trials, N_test,Delta, d):
    """
    generate #trials test sets of given dimensions using the util func in aggregating
    :return: X_test_lists, y_test_list of specified dimensions; stacked into a single numpy array (trials, N,Delta*d / 1)
    """
    X_test_list = []
    y_test_list = []

    for _ in range(trials):
        X_test,y_test =  generate_train_set(N_test,Delta,d)
        X_test_list.append(X_test)
        y_test_list.append(y_test)

    return np.stack(X_test_list,axis=0), np.stack(y_test_list,axis=0)


    
def train_and_evaluate(model, X_train, y_train, X_test_list):
    """
    trains a gpr on the trainset and then performs inference on the test sets

    :param model: the model to train
    :type model: sklearn GPR
    :param X_train: Train datapoints
    :type X_train: [type]
    :param y_train: labels 
    :type y_train: [type]
    :param Y_test_list: test sets
    :type Y_test_list: list of numpy arrays
    :return: predictions, sigma for each of the X_test sets
    :rtype: List of tuples of numpy arrays 
    """
    assert isinstance(model, GaussianProcessRegressor)

    ## train
    model.fit(X_train,y_train)

    ## evaluate 
    result_list = []
    for x_test in X_test_list:
        mu, sigma = memory_efficient_predict(model,x_test,max_size=20000)
        result_list.append((mu,sigma))
    return result_list

def soft_prediction(predictor_lists,epsilon = 1e-10):
    """
    creates the soft prediction of the bagging ensemble using the individual predictions

    :param predictor_lists: the individual predictions & sigmas
    :type predictor_lists: [[mu_i, sigma_i]]
    :return: single list with predictions
    :rtype: List of np.ndarray for each of the predicted sets
    """
    predictions = np.zeros(predictor_lists[0][0].shape[0])
    sigmas = np.zeros(predictor_lists[0][0].shape[0])
    for predictor_list in predictor_lists:
        mu,sigma = predictor_list
        mu = mu.flatten()
        predictions = predictions + ( mu / (sigma + epsilon))
        sigmas = sigmas + (1/ (sigma + epsilon))

    predictions = predictions / sigmas
    return predictions
    
def trials_soft_prediction(predictors_results,trials):
    """
    gets the predictions for a list of the evaluations for each predictor in an ensemble, for a number of trials

    :param predictors_results:  [[(mu_predictor_i,trial_j;sigma_predictor_i,trial_j) for j in range(trials)] for i in range(M)]
    :type predictors_results: [type]
    :param trials: # trials
    :type trials: [type]
    """
    prediction_list = []
    for trial in range(trials):
        predictions = [predictor[trial] for predictor in predictors_results]
        prediction = soft_prediction(predictions)
        prediction_list.append(prediction)
    return prediction_list

