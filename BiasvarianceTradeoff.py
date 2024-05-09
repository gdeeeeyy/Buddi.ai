import numpy as np
import matplotlib.pyplot as plt
import random

#a function that generates the output value for the given data point
def populationGenerator(x:int)->int:
    #generates a random value from the normal distribution with mean=0 and standard deviation=3
    r=np.random.normal(0,3)
    #the input feature value is given to the equation 2x^4 - 3x^3 + 7x^2 - 23x +8 + N(0,3) where N(0,3) is a normal distribution with mean 0 and std 3 
    y=(2*(x**4))-(3*(x**3))+(7*(x**2))-(23*(x))+8+r
    #returns the output value for the given feature value 
    return y

#an array is generated with 100 points between -5 and 5
X_init=np.array(np.linspace(-5,5,100))
#the population is generated using the populationGenerator function for the whole input feature array X and returns an array of the population
Y_init=[populationGenerator(X_init[i]) for i in range(len(X_init))]

XYtup=[]
for i in range(len(X_init)): 
    #creating a list of tuples => [(X,Y)]
    XYtup.append(tuple([X_init[i], Y_init[i]]))
#shuffles the list of tuples in place
np.random.shuffle(XYtup)

#getting the training feature values and output
X=np.array([XYtup[i][0] for i in range(len(X_init))])
Y=np.array([XYtup[i][1] for i in range(len(X_init))])

#The index for the first 80-20 split is computed
split_idx=int(X.shape[0]*0.8)

#A random sample is extracted off the XYtup, which is a list of tuples of the size of split_idx
train_data = random.sample(XYtup, split_idx)
#The data which is not in train_data but is in XYtup is utilized as the test data
test_data = [item for item in XYtup if item not in train_data]
#The X_test and Y_test value is extracted from the list of tuples test_data
X_test, Y_test = zip(*test_data)
#The X_test and Y_test value is extracted from the list of tuples test_data
X_train, Y_train = zip(*train_data) 

def coeff(X:list[float], Y:list[float], n:int)->int:
    Xtrans=[]#This array will contain the values of the X^t array
    for i in range(n+1):
        #This loop appends the value of X^i into the X^t array where i=[0,n]
        Xtrans.append(np.power(X,i))
    #The shape of Xtrans is (5, 100), so we do a transpose operation to change its shape to (100,5)
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

#generating the X values for plotting
xplot = np.linspace(-5, 5, 10000)
#the maximum degree for the polynomial to be used is initialized
deg=10
#This array contains the coefficients for each of i degree polynomial where i=[0, n]
coeffarr=[coeff(X_train, Y_train, i) for i in range(deg+1)]

#an array of the y_pred for each model is stored in an array of array of floats 

models = np.array([sum(coeffarr[i][j] * (xplot**j) for j in range(len(coeffarr[i]))) for i in range(deg+1)])

n=len(models)
txt=f"Plot that shows the performance of all the {n-1} polynomial models for the given feature input values and outputs values."
for i in range(1, deg+1):
    #This plots the curve for each of the calculated models in the given graph
    plt.plot(xplot, models[i], label=f"degree:{i} model")
#Here we plot the original data onto the graph 
plt.scatter(X_train, Y_train, label="Original Data Points", marker=".")
plt.title(f"Perform of the {n-1} models for given features")
plt.xlabel("Feature values")
plt.ylabel("Output values")
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=8)# plot description
plt.legend()
plt.show()

def meanSquaredError(y:int, y_pred:int)->int:
    #Here we calculate the mean square error with the formula sum((y-y_pred)**2)/size(y)
    return sum((y-y_pred)**2)/len(y)


def biasVarTradeoff(X_train:list[float], X_test:list[float], Y_train:list[float], Y_test:list[float], deg:int, coeffarr:list[list[float]])->tuple[list[float]]:
    bias=[]#bias array is initialized
    variance=[]#variance array is initialized
    for i in range(deg+1):
        #The predicted value of the training dataset is computed using the respective models
        y_pred_train=np.array(sum(coeffarr[i][j] * np.power(X_train, j) for j in range(len(coeffarr[i]))))
        #The predicted value of the testing dataset is computed using the models computed using the training dataset 
        y_pred_test=np.array(sum(coeffarr[i][j] * np.power(X_test,j) for j in range(len(coeffarr[i]))))
        #The mean squared error is computed for the training dataset
        yBias=meanSquaredError(Y_train, y_pred_train)
        #The mean squared error is computed for the testing dataset
        yVar=meanSquaredError(Y_test, y_pred_test)
        #The mean squared error of training dataset is stored in the bias array
        bias.append(yBias)
        #The mean squared error of testing dataset is stored in the variance array
        variance.append(yVar)
    return bias, variance

#The function biasVarTradeoff is called to return the bias and variance array for the given training and testing dataset
bias, variance=biasVarTradeoff(X_train, X_test, Y_train, Y_test, deg, coeffarr)

#The bias is plotted with respect to the degree of the models in the x axis 
plt.plot(range(0, deg+1), bias, c="r", label="Bias-Training error")
#The variance is plotted in the same graph with respect to the degree of the models in the x axis 
plt.plot(range(0, deg+1), variance, c="b", label="Variance-Testing error")
plt.legend()
plt.title('Bias and Variance Tradeoff') 
plt.xlabel('Model Complexity') 
plt.ylabel('Error') 
plt.figtext(0.5, 0.01, "Here, the bias is decreasing as the model degree increases while the variance increases", wrap=True, horizontalalignment='center', fontsize=12) # plot description
plt.show()

