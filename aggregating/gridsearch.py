import numpy as np 
import threading
from sklearn import clone 

from aggregating.utils import flatten_X, normalized_error_VT

# define your custom log for passing into each thread
def create_logger(hyperparams,log):
    def logger(x):
        print(f"logger {hyperparams}, -> {x}")
        result = list(hyperparams.values())
        result.append(x)
        log.append(result)

    return logger

def evaluate_model(base_model,hyperparams, X_train, y_train,d, DeltaT,trials,N_test, samples_generator):
    """
    Train given model on given dataset, afterwards create a new test set of size N_test and determine
    the normalized Error
    
    uses:
    - normalized_error_VT
    - Flatten_Training_Sample
    """
    print(f" {hyperparams} -> thread id = {threading.current_thread().ident}")
    model = clone(base_model)
    for k,v in hyperparams.items():
        setattr(model,k,v)
    model.fit(X_train,y_train)

    errors  = []
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
        print(errors)
    print(f"{hyperparams} -> {errors}")
    return errors
