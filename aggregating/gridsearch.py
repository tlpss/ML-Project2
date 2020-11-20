import numpy as np 
import threading
from sklearn import clone 
import traceback

from aggregating.utils import flatten_X, normalized_error_VT

# define your custom log for passing into each thread
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

def evaluate_model(base_model,hyperparams, X_train, y_train,d, DeltaT,trials,N_test, samples_generator):
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
    :return: list containing the normalized errrors
    :rtype: list of size trials
    """
 
    errors  = []
    try:
        print(f" {hyperparams} -> thread id = {threading.current_thread().ident}")
        model = clone(base_model)
        for k,v in hyperparams.items():
            setattr(model,k,v)
        model.fit(X_train,y_train)

        for trial in range(trials):
            s_test = samples_generator(N_test, d, DeltaT)
            s_test.generate_samples()
            y_test = s_test.y
            X_test = s_test.X
            S_test = s_test.S

            Flattened_X_test = flatten_X(X_test)

            V_T = y_test  
            V_0 = s_test.generate_true_V(0)
            V_0= V_0.mean()
            y_hat = model.predict(Flattened_X_test)

            error = normalized_error_VT(y_hat,V_T,V_0).item()
            print(f"{hyperparams} , {trial} -> {error}")
            errors.append(error)
        print(f"{hyperparams} -> {errors}")
        return errors

    except Exception as e:
        print(hyperparams)
        traceback.print_exc()
        return None
