#linear search - grid search

import numpy as np
import matplotlib.pyplot as plt
x=np.array([-3,-2,-1,0,1,2,3])
y=np.array([7,2,0,0,0,2,7])
b=np.arange(-1, 1, 0.001)
e_fin=[]
e_tot=[]
tab=[]

for b0 in b:
    for b1 in b:
        e=[]
        for xi,yi in zip(x,y):
            y_pred=(b0*xi)+(b1*(xi**2))
            ei=abs(y_pred-yi)
            e.append(ei)
        e_fin.append(e)
        e_sum=np.sum(e)
        e_tot.append(e_sum)
        tab.append([b0,b1,e_sum])

b0,b1,eps=[],[],[]
for i in range(len(tab)):
    b0.append(tab[i][0])
    b1.append(tab[i][1])
    eps.append(tab[i][2])
    #print(tab[i][0], tab[i][1], tab[i][2], sep="\t")
    #print("\n")

print("-------------------------------\n")
print(f"The minimum value of epsilon : {tab[np.argmin(e_tot)][2]} occurs for b0 : {tab[np.argmin(e_tot)][0]} and b1 : {tab[np.argmin(e_tot)][1]}\n")
print("-------------------------------\n")

fig=plt.figure(figsize=(7,7))
ax=fig.add_subplot(111, projection='3d')
ax.plot_trisurf(b0, b1, eps, cmap="viridis")
ax.set_xlabel("b0")
ax.set_ylabel("b1")
ax.set_zlabel("Epsilon")
ax.set_title("Error surface plot")
ax.legend()

plt.show()