#linear search - grid search

import numpy as np
import matplotlib.pyplot as plt
x=np.array([-3,-2,-1,0,1,2,3])
y=np.array([7,2,0,0,0,2,7])
b=np.arange(-1, 1, 0.001)
e_fin=[]
e_tot=[]
tab=[]
txt="estimating the optimal constants to get the smallest error estimate for the given inputs and outputs using grid search"

for b1 in b:
    for b2 in b:
        e=[]
        for xi,yi in zip(x,y):
            y_pred=(b1*xi)+(b2*(xi**2))
            ei=abs(y_pred-yi)
            e.append(ei)
        e_fin.append(e)
        e_sum=np.sum(e)
        e_tot.append(e_sum)
        tab.append([b1,b2,e_sum])

b1,b2,eps=[],[],[]
for i in range(len(tab)):
    b1.append(tab[i][0])
    b2.append(tab[i][1])
    eps.append(tab[i][2])
    #print(tab[i][0], tab[i][1], tab[i][2], sep="\t")
    #print("\n")

print("-------------------------------\n")
print(f"The minimum value of epsilon : {tab[np.argmin(e_tot)][2]} occurs for b0 : {tab[np.argmin(e_tot)][0]} and b1 : {tab[np.argmin(e_tot)][1]}\n")
print("-------------------------------\n")

fig=plt.figure(figsize=(7,7))
ax=fig.add_subplot(111, projection='3d')
ax.plot_trisurf(b1, b2, eps, cmap="viridis", label="Error suface plot with b0, b1 and Epsilon")
ax.set_xlabel("b1")
ax.set_ylabel("b2")
ax.set_zlabel("Epsilon")
ax.set_title("Error surface plot for the grid search algorithm on the given inputs and outputs to get the minimal error estimate")
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=8)
ax.legend()

plt.show()

