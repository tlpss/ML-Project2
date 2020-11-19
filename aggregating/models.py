from sklearn.base import clone, BaseEstimator
import numpy as np 

class AggregatingBaseClass(BaseEstimator):
    def __init__(self,M,predictor):
        super().__init__()
        self.M = M
        self.predictor = predictor
        self.predictors = None
        
    def _split_train_set(self,X,y):
        raise NotImplementedError
    
    def fit(self,X,y):
        print("fit")
        print(X.shape)
        self.predictors  = [clone(self.predictor)] *self.M
        X_list,y_list = self._split_train_set(X,y)
        for i in range(self.M):
            self.predictors[i].fit(X_list[i],y_list[i])
    
    def predict(self,X):
        print("predict")
        print(X.shape)
        predictions = np.zeros((X.shape[0]))
        for i in range(self.M):
            predictions = predictions + self.predictors[i].predict(X)
        predictions = predictions / self.M 
        return predictions

class SimpleSplitter(AggregatingBaseClass):
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
    def __init__(self,M,train_size_alpha,predictor):
        super(SimpleBagger,self).__init__(M,predictor)
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

class SimpleBagger(AggregatingBaseClass):
    def __init__(self,M,train_size_alpha,predictor):
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