import numpy as np 

class StochasticModelBase:
    """
    BaseClass That models a stochastic stock market with options, evaluated at T timesteps

    """
    def __init__(self,N,d,delta_Ts):
        self.N = N
        self.delta_T = delta_Ts
        self.T = len(delta_Ts) 
        self.d = d

        self.r = 0
        self.sigma_matrix  = np.identity(d)*0.2
        self.S0 = np.ones((N,d))

        self.Q = (0,1) # Q-measure of driver: iid Normal(0,1) 

        self.__reset_samples()

    def __reset_samples(self):
        """
        clear all samples and derrived values
        """
        self.X = None
        self.y = None
        self.S = None

    def generate_samples(self):
        """
        main function that generates the driver samples and computes all derrived values
        """
        self.__reset_samples()
        self._generate_driver_samples()
        self._calculate_S()
        self._calculate_y()


    def _generate_driver_samples(self):
        """
        generate the driver samples of dimension NxdxT ~ Q
        """
        self.X = np.random.normal(self.Q[0], self.Q[1], (self.N,self.d,self.T))

    def _calculate_S(self):
        """
        Calculates the Prices of the stocks using the driver samples, making use of tensor calculations to speed up the process

        :raises ValueError: if X has not been created, valueError is raised
        """
        if self.X is None:
            raise ValueError

        # One entry more because we need to store the initial value as well,
        # dimension of S = N,d,T+1
        self.S = np.zeros((self.N,self.d,self.T+1)) 
        self.S[:,:,0] = self.S0
        for (t,delta) in enumerate(self.delta_T):
            # (k,k) * (n,k,1) -> (n,k,1) where the tensor is considered as a stack of 2D matrices 
            # hence need to create the newaxis on the Nxd x-slice 
            factor = np.exp(np.matmul(self.sigma_matrix,self.X[:,:,t][:,:,np.newaxis])*np.sqrt(delta)+(self.r - np.sum(np.abs(self.sigma_matrix)**2,axis=1)/2)*delta)
            self.S[:,:,t+1] = self.S[:,:,t]*factor[:,:,1] 
    
    def _calculate_y(self):
        """
        calculates the labels (=f(X)) and stores them in self.y 

        :raises ValueError: if S was not calculated yet, throw ValueError
        """
        if self.S is None:
            raise ValueError
        
        self.y = self._calculate_f()


    def generate_true_V(self,time):
        """
        Generate V_t = E[f(X) | F_t] using nested monte carlo simulation to generate  partial, unknown trajectories for X_t+1..X_T 
        and average them to obtain an estimate on V_t  evaluated on the driver sample trajectories 

        Atm , only V_T (=f(X)) is implemented

        :param time: [description]
        :type time: [type]
        :raises NotImplementedError: only V_T is implemented atm
        :return: List of tuples (trajectory, V_t(X(j)))
        :rtype: Nx [(dxT), E[f(X)|F_t] (scalar)]
        """
        if time == 0:
            # V_0 == E[f(X)], independant of trajectories
            # so for each trajectory we add the same value
            avg_f = self._calculate_f().mean()

            return list(zip(self.X, np.full((self.X.shape[0]),avg_f).flatten())) 

        if time == self.T:
            # return list of (trajectory, V_T= f(X)) tuples
            return list(zip(self.X,self._calculate_f()))

        else:
            # requires a nested Monte Carlo Simulation for the unknown part of the trajectory t+1..T,
            #  to estimate the average of f over the remaining part of the trajectory 
            raise NotImplementedError
            

    
    
    def _calculate_f(self):
        """
        calculate the cumulative cash flow f of the trajectories

        :return: a Nx1 array containing f(X)

        :raises NotImplementedError: not implemented for this Base Class
        """
        raise NotImplementedError



class MaxCallStochasticModel(StochasticModelBase):
    """
    Extends the base Class to model a European Max call Option Portfolio
    """
    def __init__(self,N,d,deltas):
        super().__init__(N,d,deltas)
        self.K = 1 # K = S_0 in this case, which makes sense
    
    def _calculate_f(self):
        """
        European Max Call Evaluation
        """
        res = (np.max(self.S[:,:,self.T],axis=1)-self.K)
        res[res < 0] = 0
        return res*np.exp(-self.r*self.T)
        
if __name__ == "__main__":
        """
        example usage
        """
        s = MaxCallStochasticModel(10,3,[1/12,11/12])
        s.generate_samples()

        y = s.y
        X = s.X
        S = s.S
        print(S)
        V_T = s.generate_true_V(2)
        v_0 = s.generate_true_V(0)
        #print(v_0)