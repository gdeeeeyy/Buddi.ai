import numpy as np
import matplotlib.pyplot as plt
txt="This is the plot that shows the Lagrange's polynomial that perfectly passes through all the data points(Epsilon=0)"

def lagrangesPolynomial(X:list[float], Y:list[float], xi:int, n:int)->int:
    res=0.0
    for i in range(n):
        t=Y[i]#Here t is set to the ith value of Y
        for j in range(n):
            if(j!=i):
                t=t*(xi-X[j])/(X[i]-X[j])#if i!=j, we compute yi*((x-x1)...(x-xj)/(xi-xj))
        res+=t#This accumulates the value of t for all values of i and j
    return res#to return the output value of xi

X=np.array([-3,-2,-1,0,1,2,3])#An array of the inputs is defined as X
Y=np.array([7,2,0,0,0,2,7])#An array of the desired outputs is defined as Y
n=len(X)#The length of the array X is computed
Y_cap=np.array([lagrangesPolynomial(X,Y,X[i],n) for i in range(n)])#The output ycap for each values in the input array X is computed by using Lagrange's polynomial function and added to the Y_cap array
print(Y_cap)

plt.title("Applying Lagrange's polynomial to create a model that passes through all the data points")
plt.scatter(X,Y, marker="x", c="red", label="Original data points")#The actual data points are plotted in the graph
plt.plot(X,Y_cap, color="blue", label="Lagrange's polynomial line")#The lagrange's polynomial is plotted as a graph that passes through the given X points
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=8)#The string inside txt is added to the graph
plt.xlabel("Feature values")
plt.ylabel("Output values")
plt.legend()
plt.show()