import numpy as np 
import matplotlib.pyplot as plt


def visualise_stock_prices(S,DeltaT):
    """
    [summary]

    :param S: [description]
    :type S: [type]
    :param DeltaT: [description]
    :type DeltaT: [type]
    """
    d,T = S.shape
    time = np.zeros(T)
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