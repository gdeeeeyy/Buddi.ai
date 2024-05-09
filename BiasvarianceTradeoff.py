import numpy as np
import matplotlib.pyplot as plt

#a function that generates the output value for the given data point
def populationGenerator(x:int)->int:
    r=np.random.normal(0,3)#generates a random value from the normal distribution with mean=0 and standard deviation=3
    y=(2*(x**4))-(3*(x**3))+(7*(x**2))-(23*(x))+8+r#the input feature value is given to the equation 2x^4 - 3x^3 + 7x^2 - 23x +8 + N(0,3) where N(0,3) is a normal distribution with mean 0 and std 3 
    return y #returns the output value for the given feature value

X=np.array(np.linspace(-5,5,100))#an array is generated with 100 points between -5 and 5
Y=populationGenerator(X)#the population is generated using the populationGenerator function for the whole input feature array X and returns an array of the population

#getting the index for the 70-30 split for the X and Y datasets 
split_idx_train=int(X.shape[0]*0.7)

#getting the training dataset
X1_train=np.array(X[:split_idx_train])#The X array is sliced to get the training array, this is also the X^1 array
X0_train=np.array(X1_train**0)#The training array of X^0 is created
X2_train=np.array(X1_train**2)#The training array of X^2 is created
X3_train=np.array(X1_train**3)#The training array of X^3 is created
X4_train=np.array(X1_train**4)#The training array of X^4 is created
Y_train=np.array(Y[:split_idx_train])#The Y array is sliced to get the output values for the training array

#getting the validation dataset
X1_test=np.array(X[split_idx_train:])#The X array is sliced to get the output values for the training array
X0_test=np.array([X1_test**0])#The testing array of X^0 is created
X2_test=np.array([X1_test**2])#The testing array of X^2 is created
X3_test=np.array([X1_test**3])#The testing array of X^3 is created
X4_test=np.array([X1_test**4])#The testing array of X^4 is created
Y_test=np.array([Y[split_idx_train:]])#The Y array is sliced to get the output values for the testing array

#calculating the beta values
Xtrans=np.array([X0_train, X1_train, X2_train, X3_train, X4_train]) #The transpose of the X array with X^0, X^1, X^2, X^3 and X^4
X_y=np.transpose(Xtrans)#The shape of Xtrans is (5,100), therefore we take the transpose of Xtrans to get the normal X
XInv=np.linalg.inv(np.matmul(Xtrans, X_y))#We calcuate (X^t * X)^-1
calc1=np.matmul(XInv, Xtrans)#We calculate ((X^t * X)^-1)*X^t
beta=np.matmul(calc1, Y_train)#We calculate ((X^t * X)^-1)*X^t*Y to get the beta values
 
def linearModel(X1:list[float], beta:list[float])->list[float]:
    return beta[0]+(beta[1]*X1) #this function returns the values of b0+b1x^1 

def quadraticModel(X1:list[float], X2:list[float], beta:list[float])->list[float]:
    return beta[0]+(beta[1]*X1)+(beta[2]*X2)#this function returns the values of b0+b1x^1+b2x^2

def cubicModel(X1:list[float], X2:list[float], X3:list[float], beta:list[float])->list[float]:
    return beta[0]+(beta[1]*X1)+(beta[2]*X2)+(beta[3]*X3)#this function returns the values of b0+b1x^1+b2x^2+b3x^3

def quarternaryModel(X1:list[float], X2:list[float], X3:list[float], X4:list[float], beta:list[float])->list[float]:
    return beta[0]+(beta[1]*X1)+(beta[2]*X2)+(beta[3]*X3)+((beta[4]*X4))#this function returns the values of b0+b1x^1+b2x^2+b3x^3+b4x^4

def lagrangesPolynomial(X:list[float], Y:list[float], xi:int, n:int)->int:
    res=0.0
    for i in range(n):
        t=Y[i]#Here t is set to the ith value of Y
        for j in range(n):
            if(j!=i):
                t=t*(xi-X[j])/(X[i]-X[j])#if i!=j, we compute yi*((x-x1)...(x-xj)/(xi-xj))
        res+=t#This accumulates the value of t for all values of i and j
    return res#to return the output value of xi

#returns an array of the values of the complete data to plot it for each of the given data points
def models(X:list[float], Y:list[float], beta:list[float])->list[list[float]]:
    X1=X#initializing X1 to X
    X2=np.array(X1**2)#creating an array of X^2
    X3=np.array(X1**3)#creating an array of X^3
    X4=np.array(X1**4)#creating an array of X^4
    lin_model=linearModel(X1, beta)#the output array after running the input X and betas into the linearModel function is returned
    quad_model=quadraticModel(X1, X2, beta)#the output array after running the input X and betas into the quadraticModel function is returned
    cub_model=cubicModel(X1, X2, X3, beta)#the output array after running the input X and betas into the cubicModel function is returned
    quart_model=quarternaryModel(X1, X2, X3, X4, beta)#the output array after running the input X and betas into the quarternaryPolynomial function is returned
    lagranges_model=[lagrangesPolynomial(X1,Y,X1[i],len(X1)) for i in range(len(X1))]#the output array after running the input X and betas into the lagrangesPolynomial function is returned
    mod=[lin_model, quad_model, cub_model, quart_model, lagranges_model]#the mod array contains the an array of output values when feature values is passed into linear model,  quadratic model, cubic model, quarternary model and lagranges model 
    return mod

models=models(X1_train, Y_train, beta)#the array of values of all the linear, quadratic, cubic, quarternary and lagrange's model is stored inside models

#calculates the training values for the
def train(X1_train:list[float], X2_train:list[float], X3_train:list[float], X4_train:list[float], Y_train:list[float], beta:list[float])->list[float]:
    lin_model_train=linearModel(X1_train, beta)#the output array after running the input training array and betas into the linearModel function is returned
    quad_model_train=quadraticModel(X1_train, X2_train, beta)#the output array after running the input training array and betas into the quadraticModel function is returned
    cub_model_train=cubicModel(X1_train, X2_train, X3_train, beta)#the output array after running the input training array and betas into the cubicModel function is returned
    quart_model_train=quarternaryModel(X1_train, X2_train, X3_train, X4_train, beta)#the output array after running the input training array and betas into the quarternaryModel function is returned
    lagranges_model_train=[lagrangesPolynomial(X1_train,Y_train,X1_train[i],len(X1_train)) for i in range(len(X1_train))]#the output array after running the input X and betas into the lagrangesModel function is returned

    eps_lin_train = np.sum(np.abs(Y_train - lin_model_train)**2 / len(X1_train))#the MSE is calculated for the linear model
    eps_quad_train = np.sum(np.abs(Y_train - quad_model_train)**2 / len(X1_train))#the MSE is calculated for the quadratic model
    eps_cube_train = np.sum(np.abs(Y_train - cub_model_train)**2 / len(X1_train))#the MSE is calculated for the cubic model
    eps_quart_train = np.sum(np.abs(Y_train - quart_model_train)**2/ len(X1_train))#the MSE is calculated for the quarternary model
    eps_lag_train = np.sum(np.abs(Y_train - lagranges_model_train)**2 / len(X1_train))#the MSE is calculated for the lagrange's polynomial model
    op=[eps_lin_train, eps_quad_train, eps_cube_train, eps_quart_train, eps_lag_train]#This array contains the MSE for all the models
    return op

eps_bias=train(X1_train, X2_train, X3_train, X4_train, Y_train, beta)

#calculates the output values for the validation dataset
def valid(X1_test:list[float], X2_test:list[float], X3_test:list[float], X4_test:list[float], Y_test:list[float], beta:list[float])->list[float]:
    lin_model_valid=linearModel(X1_test, beta)#the output array after running the input testing array and betas into the linearModel function is returned
    quad_model_valid=quadraticModel(X1_test, X2_test, beta)#the output array after running the input testing array and betas into the quadraticModel function is returned
    cub_model_valid=cubicModel(X1_test, X2_test, X3_test, beta)#the output array after running the input testing array and betas into the cubicModel function is returned
    quart_model_valid=quarternaryModel(X1_test, X2_test, X3_test, X4_test, beta)#the output array after running the input testing array and betas into the quarternaryModel function is returned

    eps_lin_valid = np.sum((Y_test - lin_model_valid)**2/len(X1_test))#the MSE is calculated for the linear model
    eps_quad_valid = np.sum((Y_test - quad_model_valid)**2/ len(X1_test))#the MSE is calculated for the quadratic model
    eps_cube_valid = np.sum((Y_test - cub_model_valid)**2/ len(X1_test))#the MSE is calculated for the cubic model
    eps_quart_valid = np.sum((Y_test - quart_model_valid)**2/len(X1_test))#the MSE is calculated for the quarternary model
    eps=[eps_lin_valid, eps_quad_valid, eps_cube_valid, eps_quart_valid]#This array contains the MSE for all the models
    return eps

eps_variance=valid(X1_test, X2_test, X3_test, X4_test, Y_test, beta)#the epsilon values for the validation set is computed(Its gives seperate epsilon values for the linear, quadratic, cubic and biquadratic function)

#plots the training data for the models that are processed in the training data 
def plot(X_train:list[float], Y_train:list[float], models:list[float])->int:
    txt="Plot that shows the performance of the Linear, quadratic, cubic, biquadratic and lagrange's polynomial model for the given inputs and outputs"
    plt.scatter(X_train, Y_train, marker=".", label="Actual Values")
    plt.title("Model Performance Estimation(Linear, Quadratic, Cubic, Quarternary, Lagrange's polynomial)")
    plt.plot(X_train, models[0], label="Linear polynomial model")#The linear model is plotted with the given training input
    plt.plot(X_train, models[1], label="Quadratic polynomial model")#The quadratic model is plotted with the given training input
    plt.plot(X_train, models[2], label="Cubic polynomial model")#The cubic model is plotted with the given training input
    plt.plot(X_train, models[3], label="Quarternary polynomial model")#The quarternary model is plotted with the given training input
    plt.plot(X_train, models[4], label="Lagrange's polynomial model", c="r")#The lagrange's model is plotted with the given training input
    plt.xlabel("Feature values")
    plt.ylabel("Output values")
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=8)#The string inside txt is added to the graph
    plt.legend()
    plt.show()
    plt.close()
    return 1
plot(X1_train,Y_train, models)


txt="This plot presents the Bias-variance trade off for the Linear, Quadratic, Cubic, Quarternary models and a Lagrange's Polynomial"
x=[1,2,3,4,70]
plt.plot(x, eps_bias, c="r", label="Bias", marker=".")#The training bias of the models are plotted with the colour red
plt.title("Bias-Variance Trade off")
plt.xlabel("Model Complexity")
plt.ylabel("Error estimate")
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=8)#The string inside txt is added to the graph
plt.plot(x[:4], eps_variance, c="b", label="Variance", marker=".")#The testing Variance of the models are plotted with the colour blue
plt.legend()
plt.show()