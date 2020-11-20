from sklearn.base import clone, BaseEstimator
import numpy as np 

class AggregatingBaseClass(BaseEstimator):
    """
    Base Class for All Ensemble Models

    """
    def __init__(self,M,predictor):
        """

        :param M: Number of Predictors in the Ensemble
        :type M: int 
        :param predictor: the predictor model that is cloned for each instance in the ensemble
        :type predictor: sklearn.BaseEstimator
        """
        super().__init__()
        self.M = M
        self.predictor = predictor
        self.predictors = None
        
    def _split_train_set(self,X,y):
        """
        placeholder
        """
        raise NotImplementedError
    
    def fit(self,X,y):
        """
        Creates M datasets according to the _split function and trains all M predictors on their dataste

        :param X: Complete train set
        :type X: np.ndarray(N x (d*T))
        :param y: labels of the train set
        :type y: np.ndarray(N x 1 )
        """
        print("fit")
        print(X.shape)
        self.predictors  = [clone(self.predictor)] *self.M # do this here to make sure we use the latest M value (can be set dynamically)
        X_list,y_list = self._split_train_set(X,y)
        for i in range(self.M):
            self.predictors[i].fit(X_list[i],y_list[i])
    
    def predict(self,X):
        """
        Predicts the value of the set X using the unweighted average of the M predictors ("hard voting" for regression)

        :param X: set of datapoints for which to make a prediction
        :type X: np.ndarray(N x (d*T))
        :return: Predictions 
        :rtype: np.ndarray(N x 1)
        """
        print("predict")
        print(X.shape)
        predictions = np.zeros((X.shape[0]))
        # unweighted average
        for i in range(self.M):
            predictions = predictions + self.predictors[i].predict(X)
        predictions = predictions / self.M 
        return predictions

class SimpleSplitter(AggregatingBaseClass):
    """
    Simples case of aggregation, split the trainset evenly over the M predictors

    """
    def __init__(self,M,predictor):
        super(SimpleSplitter,self).__init__(M,predictor)
    
    def _split_train_set(self,X,y):
        n = X.shape[0]// self.M
        X_list = []
        y_list = []
        for i in range(self.M-1):
            X_list.append(X[n*i:n*(i+1)])
            y_list.append(y[n*i:n*(i+1)])
        X_list.append(X[(self.M-1)*n:])
        y_list.append(y[(self.M-1)*n:])
        return X_list, y_list

class SimplePaster(AggregatingBaseClass):
    """
    Pasting Aggregation, all datasets are drawn from the original dataset using replacement between the different sets
    but not within the sets.

    e.g. [1,2,3,4,5] -> [1,4,5], [1,2,4]

    """
    def __init__(self,M,train_size_alpha,predictor):
        """
        :param train_size_alpha: relative size of each trainset w.r.t. the original trainset
        :type train_size_alpha: float in range (0,1]
        """
        super(SimplePaster,self).__init__(M,predictor)
        self.train_size_alpha = train_size_alpha
        


class SimpleBagger(AggregatingBaseClass):
    """
    Bagging Aggregation, all datasets are drawn from the original dataset using replacement between the different sets
    AND within the sets.

    e.g. [1,2,3,4,5] -> [1,4,5], [1,3,3]

    """
    def __init__(self,M,train_size_alpha,predictor):
        """    def _split_train_set(self,X,y):
        n = round(X.shape[0]*self.train_size_alpha)
        X_list = []
        y_list = []
        for i in range(self.M):
            indices = np.random.choice(X.shape[0],size=n,replace=False)
            X_list.append(X[indices])
            y_list.append(y[indices])
        return X_list, y_list
        :param train_size_alpha: relative size of each trainset w.r.t. the original trainset
        :type train_size_alpha: float in range (0,1]
        """
        super(SimpleBagger,self).__init__(M,predictor)
        self.train_size_alpha = train_size_alpha
     
    def _split_train_set(self,X,y):
        n = round(X.shape[0]*self.train_size_alpha)
        X_list = []
        y_list = []
        for i in range(self.M):
            indices = np.random.choice(X.shape[0],size=n,replace=True)
            X_list.append(X[indices])
            y_list.append(y[indices])
        return X_list, y_list

class SoftAggregatingBaseClass(AggregatingBaseClass):
    """
    Performs Soft aggregation voting using the explicit sigmas that come with each prediction.
    Each prediction is weighted using 1/ (sigma + epsilon) and the total is normalized using the sum over these factors. 

    """

    def __init__(self, M, predictor, epsilon = 1e-8):
        self.epsilon = epsilon
        super(SoftAggregatingBaseClass,self).__init__(M,predictor)
    
    def predict(self,X):
        print(" soft predict")
        print(X.shape)
        predictions = np.zeros((X.shape[0]))
        sigmas = np.zeros(X.shape[0])
        for i in range(self.M):
            mu,sigma = self.predictors[i].predict(X,return_std=True)
            predictions = predictions + ( mu / (sigma + self.epsilon))
            sigmas = sigmas + (1/ (sigma + self.epsilon))

        predictions = predictions / sigmas
        return predictions

class SoftPaster(SoftAggregatingBaseClass):
    """
    Pasting Aggregation, all datasets are drawn from the original dataset using replacement between the different sets
    but not within the sets.

    e.g. [1,2,3,4,5] -> [1,4,5], [1,2,4]

    """
    def __init__(self,M,train_size_alpha,predictor):
        """
        :param train_size_alpha: relative size of each trainset for the M predictors
        :type train_size_alpha: float

        """
        super(SoftPaster,self).__init__(M,predictor)
        self.train_size_alpha = train_size_alpha


    def _split_train_set(self,X,y):
        n = round(X.shape[0]*self.train_size_alpha)
        X_list = []
        y_list = []
        for i in range(self.M):
            indices = np.random.choice(X.shape[0],size=n,replace=False)
            X_list.append(X[indices])
            y_list.append(y[indices])
        return X_list, y_list

class SoftBagger(SoftAggregatingBaseClass):
    """
    Soft Bagging Aggregation, all datasets are drawn from the original dataset using replacement between the different sets
    AND within the sets. Weighted averaging using the sigmas. 

    e.g. [1,2,3,4,5] -> [1,4,5], [1,3,3]

    """
    def __init__(self,M,train_size_alpha,predictor):
        """
        :param train_size_alpha: relative size of each trainset for the M predictors
        :type train_size_alpha: float

        """
        super(SoftBagger,self).__init__(M,predictor)
        self.train_size_alpha = train_size_alpha


    def _split_train_set(self,X,y):
        n = round(X.shape[0]*self.train_size_alpha)
        X_list = []
        y_list = []
        for i in range(self.M):
            indices = np.random.choice(X.shape[0],size=n,replace=True)
            X_list.append(X[indices])
            y_list.append(y[indices])
        return X_list, y_list



    