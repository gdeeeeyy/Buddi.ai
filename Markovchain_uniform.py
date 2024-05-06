import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

darts = [0, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
pi_caps=[]
txt="This is a Monte Carlo simulation that is used to estimate the value of pi by throwing darts onto an unit square which contains an unit circle, so the probability of the darts falling onto the circle will be pi/4 and the value of pi will be 4 times the probabily of darts falling into the unit circle. This graph shows the final pi value for total number of darts thrown onto the unit square"

darts_inside, tot_darts=0,0
for k in range(len(darts)-1):
    for i in range(darts[k], darts[k+1]):
        xi=np.random.uniform(-1/2, 1/2)
        yi=np.random.uniform(-1/2, 1/2)
        dist=math.sqrt(xi**2+yi**2)
        if(dist<=0.5):
            darts_inside+=1
        tot_darts+=1
    pi_cap=4*(darts_inside/tot_darts)
    print(pi_cap)
    pi_caps.append(pi_cap)
print(pi_caps)

plt.xscale("log")
plt.xlabel("Total number of darts")
plt.ylabel("Estimated pi value")
plt.axhline(y=math.pi, color="k", linestyle="--")
plt.title("Monte carlo simutation to estimate value of pi(using uniform random samples)")
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=8)
plt.plot(darts[1:], pi_caps, marker="x", label="Estimated value of pi")
plt.legend()
plt.show()

#smoothing
#normaldist - randn (normal distribution)