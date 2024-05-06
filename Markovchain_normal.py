import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

darts = [0, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
pi_capsn=[]
txt="This is a Monte Carlo simulation that is used to estimate the value of pi by throwing darts onto an unit square which contains an unit circle, so the probability of the darts falling onto the circle will be pi/4 and the value of pi will be 4 times the probabily of darts falling into the unit circle. This graph shows the final pi value for total number of darts thrown onto the unit square"
std=3

darts_inside, tot_darts=0,0
for k in range(len(darts)-1):
    for i in range(darts[k], darts[k+1]):
        xi=np.random.normal(0, std)
        while(xi>0.5 or xi<-0.5):
            xi=np.random.normal(0,std)
        yi=np.random.normal(0, std)
        while(yi>0.5 or yi<-0.5):    
            yi=np.random.normal(0,std)
        distn=math.sqrt(xi**2+yi**2)
        if(distn<=0.5):
            darts_inside+=1
        tot_darts+=1
    pi_capn=4*(darts_inside/tot_darts)
    pi_capsn.append(pi_capn)
    print(pi_capn)
print(pi_capsn)

plt.xscale("log")
plt.xlabel("Total number of darts")
plt.ylabel("Estimated pi value")
plt.axhline(y=math.pi, color="k", linestyle="--") 
plt.title("Monte carlo simutation to estimate value of pi(using normal samples)")
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=8)
plt.plot(darts[1:], pi_capsn, marker="x", label="Estimated value of pi")
plt.legend()
plt.show()
