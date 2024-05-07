import numpy as np
import matplotlib.pyplot as plt

def normal_func(x, mean, std):
    y = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2)) 
    return y


def same_mean(m):
    s1,s2,s3=1,1.5,2
    xsm1=np.arange(m-3*s1, m+3*s1, 0.1)
    xsm2=np.arange(m-3*s2, m+3*s2, 0.1)
    xsm3=np.arange(m-3*s3, m+3*s3, 0.1)
    ysm1=normal_func(xsm1,m,s1)
    ysm2=normal_func(xsm2,m,s2)
    ysm3=normal_func(xsm3,m,s3)
    plt.plot(xsm1, ysm1, label=f"Normal curve with mean {m} and standard deviation {s1}")
    plt.plot(xsm2, ysm2, label=f"Normal curve with mean {m} and standard deviation {s2}")
    plt.plot(xsm3, ysm3, label=f"Normal curve with mean {m} and standard deviation {s3}")
    plt.legend()
    plt.title("Normal curve with same mean and different standard deviation")
    plt.show()

def same_std(s):
    m1,m2,m3=1,1.5,2
    xsm1=np.arange(m1-3*s, m1+3*s, 0.1)
    xsm2=np.arange(m2-3*s, m2+3*s, 0.1)
    xsm3=np.arange(m3-3*s, m3+3*s, 0.1)
    ysm1=normal_func(xsm1,m1,s)
    ysm2=normal_func(xsm2,m2,s)
    ysm3=normal_func(xsm3,m3,s)
    plt.plot(xsm1, ysm1, label=f"Normal curve with mean {m1} and standard deviation {s}")
    plt.plot(xsm2, ysm2, label=f"Normal curve with mean {m2} and standard deviation {s}")
    plt.plot(xsm3, ysm3, label=f"Normal curve with mean {m3} and standard deviation {s}")
    plt.legend()
    plt.title("Normal curve with different mean and same standard deviation")
    plt.show()


same_std(1)

#smds
#dmss