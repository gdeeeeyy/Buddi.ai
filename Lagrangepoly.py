import numpy as np
import matplotlib.pyplot as plt
txt="This is the plot that shows the Lagrange's polynomial that perfectly passes through all the data points(Epsilon=0)"

def lagrange(X, Y, xi, n):
    res=0.0
    for i in range(n):
        t=Y[i]
        for j in range(n):
            if(j!=i):
                t=t*(xi-X[j])/(X[i]-X[j])
        res+=t
    return res

X=np.array([-3,-2,-1,0,1,2,3])
Y=np.array([7,2,0,0,0,2,7])
n=len(X)
Y_cap=np.array([lagrange(X,Y,X[i],n) for i in range(n)])
print(Y_cap)

plt.title("Applying Lagrange's polynomial to create a model that passes through all the data points")
plt.scatter(X,Y, marker="x", c="red", label="Original data points")
plt.plot(X,Y_cap, color="blue", label="Lagrange's polynomial line")
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=8)
plt.xlabel("Feature values")
plt.ylabel("Output values")
plt.legend()
plt.show()