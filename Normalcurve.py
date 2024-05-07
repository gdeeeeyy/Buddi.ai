import numpy as np
import matplotlib.pyplot as plt

txt1="This plots in the graph have the same position of the peak but different heights and widths as the standard deviation changes and that controls the width and height of the peak"
txt2="This plots in the graph have the same height and width but the position of the peak changes for different means as mean controls the position of the peak"

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
    plt.plot(xsm1, ysm1, label=f"Normal curve with μ={m} and δ={s1}")
    plt.plot(xsm2, ysm2, label=f"Normal curve with μ={m} and δ={s2}")
    plt.plot(xsm3, ysm3, label=f"Normal curve with μ={m} and δ={s3}")
    plt.legend()
    plt.title("Normal curve with same mean and different standard deviation")
    plt.figtext(0.5, 0.01, txt1, wrap=True, horizontalalignment='center', fontsize=8)
    plt.show()

def same_std(s):
    m1,m2,m3=1,1.5,2
    xsm1=np.arange(m1-3*s, m1+3*s, 0.1)
    xsm2=np.arange(m2-3*s, m2+3*s, 0.1)
    xsm3=np.arange(m3-3*s, m3+3*s, 0.1)
    ysm1=normal_func(xsm1,m1,s)
    ysm2=normal_func(xsm2,m2,s)
    ysm3=normal_func(xsm3,m3,s)
    plt.plot(xsm1, ysm1, label=f"Normal curve with  μ={m1} and δ={s}")
    plt.plot(xsm2, ysm2, label=f"Normal curve with μ={m2} and δ={s}")
    plt.plot(xsm3, ysm3, label=f"Normal curve with μ={m3} and δ={s}")
    plt.figtext(0.5, 0.01, txt2, wrap=True, horizontalalignment='center', fontsize=8)
    plt.legend()
    plt.title("Normal curve with different mean and same standard deviation")
    plt.show()

same_mean(0)
same_std(1)



#smds
#dmss