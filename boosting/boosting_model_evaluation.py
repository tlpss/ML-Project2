import numpy as np 
import threading
from sklearn import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from aggregating.utils import *
import traceback

def evaluate_boosting(X_train, y_train, X_test, y_test, V_0_train, Max_Iter, min_error, early_stop, learning_rate, epsilon, sample_size, V_0_test= None, logger = None):
    np.random.seed(2020)
    error_going_up = 0
    N_train = len(y_train)
    N_test = len(y_test)
    train_prediction = np.zeros(N_train)
    test_prediction = np.zeros(N_test)
    previous_y_hat = np.zeros(N_train)
    previous_y_test_hat = np.zeros(N_test)
    models = []
    train_errors = []
    test_errors = []
    
    current_residual = np.full(len(y_train), np.mean(y_train, axis=0))
    y_hat_test = np.full(len(y_test), np.mean(y_train, axis=0))

   
    
    for i in range(Max_Iter):
        model = create_GPR(N_train)
        
        indices = np.random.choice(X_train.shape[0],size=sample_size,replace=False)
       
        model.fit(X_train[indices], y_train[indices] - current_residual[indices])
        models.append(model)
        
        new_train_predictor = model.predict(X_train)
        new_test_predictor = model.predict(X_test)
        
        current_residual += learning_rate * new_train_predictor
        
        y_hat_train = current_residual 
        
        y_hat_test += learning_rate * new_test_predictor
        
        #Evaluating TrainSet error
        
        train_errors.append(normalized_error_VT(y_hat_train,y_train,V_0_train).item())
        
        ##Evaluating TestSet error
        
        test_errors.append(normalized_error_VT(y_hat_test,y_test,V_0_test).item())
    
        if (test_errors[-1] < min_error[0]):
            if (( np.abs(min_error[0] - test_errors[-1]) < epsilon)):
                models = models[:min_error[1]+1]
                test_errors = test_errors[:min_error[1]+1]
                best_predictor = y_hat_test
                print("For iteration number {}, the boosting stops as the error isn't decraesing enough anymore with epsilon: {}, test error: {}".format(i, epsilon, min_error[0]))
                logger.info("For iteration number {}, the boosting stops as the error isn't decraesing enough anymore with epsilon: {}, test error: {}".format(i, epsilon, min_error[0]))
                return train_errors, test_errors, min_error, best_predictor
            
            min_error = (test_errors[-1], i)
            error_going_up = 0
            print('For iteration number {}, the test error decreased , test error : {} '.format(i, min_error[0]))
            logger.info("For iteration number {}, the test error decreased , test error : {} ".format(i, min_error[0]))
        else:
            error_going_up += 1
            print('For iteration number {}, the test error increased , test error : {} '.format(i, min_error[0]))
            logger.info("For iteration number {}, the test error increased , test error : {} ".format(i, min_error[0]))
        
            if  (i==(Max_Iter-1)):
                logger.info(f"Max_Iter {Max_Iter} reached")
                models = models[:min_error[1]+1]
                test_errors = test_errors[:min_error[1]+1]
                return train_errors, test_errors, min_error, best_predictor
        
            elif (error_going_up == early_stop ):
                logger.info(f"Early_Stop {early_stop} reached")
                models = models[: -(early_stop)]
                test_errors = test_errors[:-(early_stop)]
                return train_errors, test_errors, min_error, best_predictor
                break #early stopping
                
         
        
    return train_errors, test_errors, min_error, best_predictor

def evaluate_boosting_1(X_train, y_train, X_test, y_test, V_0_train, Max_Iter, min_error, early_stop, learning_rate, epsilon, sample_size, V_0_test= None, logger = None):
    np.random.seed(2020)
    error_going_up = 0
    N_train = len(y_train)
    N_test = len(y_test)
    train_prediction = np.zeros(N_train)
    test_prediction = np.zeros(N_test)
    previous_y_hat = np.zeros(N_train)
    previous_y_test_hat = np.zeros(N_test)
    models = []
    train_errors = []
    test_errors = []
    
    y_hat_train = 0.83*fX_2
    y_hat_test = 0.83*fX_1
    
    for i in range(Max_Iter):
        #Evaluating TrainSet error
        
        train_errors.append(normalized_error_VT(y_hat_train,y_train,V_0_train).item())
        
        ##Evaluating TestSet error
        
        test_errors.append(normalized_error_VT(y_hat_test,y_test,V_0).item())
        
        model = GaussianProcessRegressor(kernel)
    
        indices = np.random.choice(X_train.shape[0],size=sample_size,replace=False)
       
        model.fit(X_train[indices], y_train[indices] - y_hat_train[indices])
        models.append(model)
        
        new_train_predictor = model.predict(X_train)
        new_test_predictor = model.predict(X_test)
        
        
        y_hat_train += learning_rate * new_train_predictor
        
        y_hat_test += learning_rate * new_test_predictor
        
        
        if (test_errors[-1] < min_error[0]):
            if (( np.abs(min_error[0] - test_errors[-1]) < epsilon)):
                models = models[:min_error[1]+1]
                test_errors = test_errors[:min_error[1]+1]
                print("For iteration number {}, the boosting stops as the error isn't decraesing enough anymore, test error: {}".format(i, min_error[0]))
                logger.info("For iteration number {}, the boosting stops as the error isn't decraesing enough anymore, test error: {}".format(i, min_error[0]))
                return train_errors, test_errors, min_error
            
            min_error = (test_errors[-1], i)
            error_going_up = 0
            print('For iteration number {}, the test error decreased , test error : {} '.format(i, min_error[0]))
            logger.info("For iteration number {}, the test error decreased , test error : {} ".format(i, min_error[0]))
        else:
            error_going_up += 1
            print('For iteration number {}, the test error increased , test error : {} '.format(i, min_error[0]))
            logger.info("For iteration number {}, the test error increased , test error : {} ".format(i, min_error[0]))
        
            if  (i==(Max_Iter-1)):
                logger.info(f"Max_Iter {Max_Iter} reached")
                models = models[:min_error[1]+1]
                test_errors = test_errors[:min_error[1]+1]
                return train_errors, test_errors, min_error
        
            elif (error_going_up == early_stop ):
                logger.info(f"Early_Stop {early_stop} reached")
                models = models[: -(early_stop)]
                test_errors = test_errors[:-(early_stop)]
                return train_errors, test_errors, min_error
                break #early stopping
                
         
        
    return train_errors, test_errors, min_error

