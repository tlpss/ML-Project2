import numpy as np 
import matplotlib.pyplot as plt


def visualise_stock_prices(S,DeltaT,start_time = 0):
    """
    Plots stock prices 

    :param S: Stock prices of a bundle of stocks
    :type S: d x T np 2D array
    :param DeltaT: array of evaluation moments for stock prices
    :type DeltaT: list of len T-1
    :param start_time: time relative to which the DeltaT array times are expressed, defaults to 0
    :type start_time: int, optional
    """
    d,T = S.shape
    time = np.ones(T)*start_time
    print(time)
    time[1:] = [time[i-1] + DeltaT[i-1] for i in range(1,T)]
    print(time)
    plt.figure()

    for index in range(S.shape[0]): 
        plt.plot(time,S[index,:],"-o", label=f"stock {index}")
    
    plt.xlabel("time")
    plt.ylabel("stock value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    S = np.array([[1.,         1.05223331 ,1.12558271],
  [1.       ,  0.93761344, 0.83247555],
  [1.      ,   0.93343528 ,0.77710408]])
    DeltaT = [1/12,11/12]
    visualise_stock_prices(S,DeltaT)