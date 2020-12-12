import numpy as np 
import threading
from sklearn import clone 
import traceback

from aggregating.utils import flatten_X, normalized_error_VT

# define  custom log for passing into each thread for a threadpool
def create_logger(hyperparams,log):
    """
    creates async log call

    :param hyperparams: dictionnary of the hyperparams
    :type hyperparams: dict{str: value}
    :param log: list to append the results to
    :type log: list[ [dict.values, results]]
    """
    def logger(x):
        print(f"logger {hyperparams}, -> {x}")
        result = list(hyperparams.values())
        result.append(x)
        log.append(result)

    return logger

def evaluate_model(base_model,hyperparams, X_train, y_train,d, DeltaT,trials,N_test, samples_generator,V_0 = None, logger = None,random_seeds = None):
    """
    Sets hyperparameters of the model, trains it first and evaluates it afterwards. 
    Used in GridSearch settings

    :param base_model: model to used
    :type base_model: sklearn model
    :param hyperparams: dict of hyperparameters, containing their exact name as key 
    :type hyperparams: dict{str: value}
    :param X_train: Training set
    :type X_train: np.ndarray(N x (d*T))
    :param y_train: labels of training set
    :type y_train: np.ndarray( N x 1 )
    :param d: number of stocks 
    :type d: integer
    :param DeltaT: list of the delta Ts between the driver moments
    :type DeltaT: list of length T
    :param trials: number of times to evaluate the model
    :type trials: integer
    :param N_test: number of samples for each evaluation
    :type N_test: integer
    :param samples_generator: stochastic generator used to generate test samples 
    :type samples_generator: StochasticModelBase
    :param V_0: V_0 value, could be obtained using larger set to improve accuracy of the results 
    :type V_0: optional float
    :param logger: Optional, logger for debug information 
    :param seeds: Optional, list of seeds (in size of trials),
                so that the same seed is used to generate the testset over multiple instances of the gridsearch
    :return: list containing the normalized errrors
    :rtype: list of size trials
    """
 
    errors  = []
    try:
        if logger:
            logger.debug(f" {hyperparams} -> thread id = {threading.current_thread().ident}")
        else:
            print(f" {hyperparams} -> thread id = {threading.current_thread().ident}")
        # create the model and set the hyperparams
        model = clone(base_model)
        for k,v in hyperparams.items():
            setattr(model,k,v)
        # train the model 
        model.fit(X_train,y_train)

        # evaluate the model 
        for trial in range(trials):
            if random_seeds:
                np.random.seed(random_seeds[trial])
            s_test = samples_generator(N_test, d, DeltaT)
            s_test.generate_samples()
            y_test = s_test.y
            X_test = s_test.X
            S_test = s_test.S

            Flattened_X_test = flatten_X(X_test)

            V_T = y_test  
            if V_0 is None:
                V_0 = s_test.generate_true_V(0)
                V_0= V_0.mean()
            y_hat = model.predict(Flattened_X_test)

            error = normalized_error_VT(y_hat,V_T,V_0).item()
            if logger:
                logger.debug(f"{hyperparams} , {trial} -> {error}")
            else:
                print(f"{hyperparams} , {trial} -> {error}")
            # add normalized error to the list 
            errors.append(error)
        if logger:
            logger.info(f"{hyperparams} -> {errors}")
        else:   
            print(f"{hyperparams} -> {errors}")
        return errors

    except Exception as e:
        if logger:
            logger.warn(traceback.format_exc())
        else:
            traceback.print_exc()
        return None
