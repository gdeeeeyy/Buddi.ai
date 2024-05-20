import numpy as np
import matplotlib.pyplot as plt

def I(x:chr)->int:
    #return 0 if x is 'G' else, returns 1
    return 0 if x=='G' else 1

def Sigmoid(x:float, β0:float, β1:float)->float:
    return 1/(1+np.exp(-β0-β1*x))

def δ(x:float, threshold:float)->chr:
    #returns 'B' if x>0.55 else 'G'
    return 'B' if x>threshold else 'G'

def gradientDesc(X:list[float], Y:list[float], beta:list[float], a:float, learning_rate:float)->list[list[float]]:
    b0, b1=beta[0], beta[1]
    for _ in range(10000):
        y_pred=Sigmoid(X, b0, b1)
        dedb0=np.mean(-2*(Y-y_pred)*((y_pred)**2)*(np.exp(y_pred)))
        dedb1=np.mean(-2*(Y-y_pred)*((y_pred)**2)*(np.exp(y_pred))*X)
        b0-=learning_rate*dedb0
        b1-=learning_rate*dedb1
    return b0, b1

def main():
    #The array with all data points are given
    X=np.array([1,2,3,4,5,-3,-4,-5,-6])
    #The array with the corresponding values for each corresponding data point is given
    Y=np.array(['B','B','B','B','B', 'G','G','G','G'])
    #The boolean value for each label is returned inside Y_bool
    Y_bool=[I(yi) for yi in Y]
    #The β0 and β1 values are calculated through closed form method
    β0, β1=gradientDesc(X, Y_bool, [0,0], 1, 0.01)
    print(β0, β1)
    #The predicted values for each data point is computed
    Y_pred=β0+β1*X
    #The threshold is set as the average of all Y values
    threshold=np.mean(Y_bool)
    #The value of each data point is converted to its corresponding label finally 
    Y_fin=[δ(yi, threshold) for yi in Y_pred]
    #The array of data pts with label "B" is used in X_label1
    X_label1=np.array([X[idx] for idx in range(len(Y)) if Y[idx]=="B"])
    #The array of data pts with label "G" is used in X_label2
    X_label2=np.array([X[idx] for idx in range(len(Y)) if Y[idx]=="G"])
    #The xSeries and ySeries value is returned for the decision boundary
    print(f"The values after applying a regressor on the labels and given points are : {list(Y_pred)}")
    print(f"The values after finding the labels using the Discriminator function for the regressor values are : {Y_fin}")
    plt.title("Classfication using a linear regression")
    plt.axhline(y=0, c="k")
    plt.axvline(x=0, c="k")
    plt.scatter(X_label1, np.zeros(shape=X_label1.shape), marker="x", c="r", label="Data points with label B")
    plt.scatter(X_label2, np.zeros(shape=X_label2.shape), marker="o", c="b", label="Data points with label G")
    plt.scatter(X, Y_pred, marker="+", c="orange", label="The predicted values for X using regression")
    plt.figtext(0.5, 0.01, "We have executed binary classification using a linear regression by converting the labels into boolean values(0,1)", wrap=True, horizontalalignment='center', fontsize=12)
    plt.plot(X, Y_pred, c="g", label="The regression line")
    plt.legend()
    #plt.show()

if __name__=="__main__":
    main()