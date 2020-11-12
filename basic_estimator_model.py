from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class EstimatorModelBase:
    def __init__(self, kernel):
        self.kernel = kernel
        self.gpr = GaussianProcessRegressor(kernel)
        
    def fit(self,X,y):
        return self.gpr.fit(X,y)
    
    def predict(self,X):
        pass
    
    def optimal_hyperparameters(self):
        return self.kernel.theta
    
    def _predict_fX(self, X):
        
        self.fX = self.gpr.predict(X)
        

    def _predict_VXt(self, X, t, aplha, betha, d):
        pass
        