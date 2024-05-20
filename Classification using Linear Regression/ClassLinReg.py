import numpy as np
import matplotlib.pyplot as plt

def I(x:chr)->int:
    #return 0 if x is 'G' else, returns 1
    return 0 if x=='G' else 1

def coeff(X:list[int], Y:list[int])->list[list[float]]:
    #This array will contain the values of the X^t array
    Xtrans=[X**0, X]
    #The shape of Xtrans is (2, 1), so we do a transpose operation to change its shape to (1,2)
    Xtrans=np.array(Xtrans)
    Xnew=np.transpose(Xtrans)
    #This equation is computed below beta=(X^t * X)^-1 * X^t * Y
    #We compute X^t * X
    XtX=np.matmul(Xtrans, Xnew)
    #we compute X^t * Y
    XtY=np.matmul(Xtrans, Y)
    #We compute (X^t * X)^-1
    Xinv=np.linalg.inv(XtX)
    #We compute beta=(X^t * X)^-1 * X^t * Y
    beta=np.matmul(Xinv, XtY)
    return beta 

def δ(x:float, threshold:float)->chr:
    #returns 'B' if x>0.55 else 'G'
    return 'B' if x>threshold else 'G'

def plotNormal(X:list[float], Y:list[float]):
    Y=list(Y)
    #The base point from the plot is taken
    p1=[X[Y.index(min(Y))], min(Y)]
    #The max point from the plot is taken
    p2=[X[Y.index(max(Y))], max(Y)]
    #caclulating the slope of the line
    slope=(p2[1]-p1[1])/(p2[0]-p1[0])
    #The slope of the normal line is (-1/slope)
    normSlop=float(-1/slope)
    #calculating intercept of normal line using y=(-1/slope)x +c
    c=np.mean(Y)
    #generates an numpy array with the minimum point and the max point
    xSeries=np.array([min(p1[0], p2[0]), max(p1[0], p2[0])])
    #calculating the y values of the normal line y_normal=(-1/slope)*X+c
    ySeries=(normSlop*xSeries)+c
    return xSeries, ySeries

def meanCountError(Y:list[int], Y_pred:list[int])->int:
    c=0
    for i in range(len(Y)):
        if(Y[i]!=Y_pred[i]):
            c+=1
    return c

def threshVsError(X:list[float], Y:list[int], beta:list[float]):
    thresholds=np.arange(X.min(), X.max(), 0.5)
    β0,β1=beta[0], beta[1]
    eps=[]
    for i in range(len(thresholds)):
        Y_pred=[I(δ(β0+β1*X[j], thresholds[i])) for j in range(len(X))]
        eps.append(meanCountError(Y, Y_pred))
    plt.title("Threshold vs Mean Count Error plot")
    plt.plot(thresholds, eps, label="Relationship between threshold and mean count error")
    plt.xlabel("Threshold")
    plt.ylabel("Mean Count Error")
    plt.figtext(0.5, 0.01, "This plot shows the relationships between threshold and the mean count error for each threshold", wrap=True, horizontalalignment='center', fontsize=12)
    plt.legend()
    plt.show()
    plt.close()

def main():
    #The array with all data points are given
    X=np.array([1,2,3,4,5,-3,-4,-5,-6])
    #The array with the corresponding values for each corresponding data point is given
    Y=np.array(['B','B','B','B','B', 'G','G','G','G'])
    #The boolean value for each label is returned inside Y_bool
    Y_bool=[I(yi) for yi in Y]
    #The β0 and β1 values are calculated through closed form method
    β0, β1=coeff(X, Y_bool)
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
    xSeries, y_Series=plotNormal(X, Y_pred)
    #This function plots the classification threshold vs the error graph
    threshVsError(X, Y_bool, [β0, β1])
    print(f"The values after applying a regressor on the labels and given points are : {list(Y_pred)}")
    print(f"The values after finding the labels using the Discriminator function for the regressor values are : {Y_fin}")
    plt.title("Classfication using a linear regression")
    plt.axhline(y=0, c="k")
    plt.axvline(x=0, c="k")
    plt.scatter(X_label1, np.zeros(shape=X_label1.shape), marker="x", c="r", label="Data points with label B")
    plt.scatter(X_label2, np.zeros(shape=X_label2.shape), marker="o", c="b", label="Data points with label G")
    plt.scatter(X, Y_bool, marker="+", c="orange", label="The predicted values for X using regression")
    plt.figtext(0.5, 0.01, "We have executed binary classification using a linear regression by converting the labels into boolean values(0,1)", wrap=True, horizontalalignment='center', fontsize=12)
    plt.plot(X, Y_pred, c="g", label="The regression line")
    plt.plot(xSeries, y_Series, c="r", label="The normal line/decision boundary", ls="dashdot")
    plt.ylim(-6.9,6.9)
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()