import numpy as np
import matplotlib.pyplot as plt
import random

#Initializing a curried function linearPolyModel with a beta array as a parameter
def linearPolyModel(beta:list[float]):
    #A function _impl is defined inside linearPolyModel to curry the function with x as input
    def _impl(x:float)->float:
        #calcs the b0+b1*x value for the given x
        return beta[0] + (beta[1]*x)
    #returns the value of the function _impl
    return _impl

def meanSquaredError(y:float, y_pred:float)->float:
    #Here we calculate the mean square error with the formula sum((y-y_pred)**2)/size(y)
    return np.mean((y-y_pred)**2)


def gradDescent(X:list[float], Y:list[float], X_test:list[float], Y_test:list[float], Y_pred:float, beta:list[float], learning_rate:float)->list[list[float]]:
    #an array to add a tuple with b0 and b1 value for each epoch, it is initialized with the initial b0 and b1 values
    bUpdate=[[beta[0], beta[1]]]
    #Epoch counter initlialized to 0
    epochs=0
    #The Yvals array is initialised with the first predicted value of Y with random b0 and b1 generated, that is closer to 0
    Yvals=[Y_pred]
    #the eps list is initalized, this will contain all the mse values for all the epochs
    eps_train=[meanSquaredError(Y_pred, Y)]
    eps_test=[meanSquaredError(Y_pred, Y_test)]
    #The flag is set to 1
    flag=1
    #This loop executes until the flag value becomes false
    while(flag):
        idx=np.random.randint(0, len(X))
        xi=X[idx]
        yi=Y[idx]
        #The b0 and b1 value of the last inserted value of the element is used as b0 and b1(changes for each epoch)
        b0,b1=bUpdate[-1][0],bUpdate[-1][1] 
        yNew=b0+b1*xi
        #partial derivative of error func wrt b0 is computed(db/db0=(2y-y_pred)(-1))
        dedb0=np.mean(-2*(yi- yNew)) 
        #partial derivative of error func wrt b1 is computed(db/db0=(2y-y_pred)(-x))
        dedb1=np.mean(2*((yi-yNew)* (-xi)))
        #The value of b0 is updated for each epoch, each value of b0 and b1
        #b0 old= b0 new+ learning_rate*(de/db0)
        b0-=(learning_rate*dedb0)
        #b1 old= b1 new+ learning_rate*(de/db1)
        b1-=(learning_rate*dedb1)
        # This array contains the value of b0 and b1 for each iteration
        betArr=[b0, b1] 
        #The curried function is called with the beta array
        Y_pred_new=linearPolyModel(betArr)
        Y_pred_test=linearPolyModel(betArr)
        #The curryied function is used to generate the value of Yprednew for each value of X 
        Yprednew=Y_pred_new(X)
        #The updated b0 and b1 value is appened into bUpdate as an array -> [b0, b1]
        bUpdate.append(betArr) 
        #The epochs is counted for each iteration
        epochs+=1
        #The mean squared error for the previos Y value and the present Y value is calculated
        epsi=meanSquaredError(Yprednew[idx], Yvals[-1])
        #The mse is then appended to the eps list
        eps_train.append(meanSquaredError(Y, Yprednew))
        eps_test.append(meanSquaredError(Y_test, Y_pred_test(X_test)))
        #The flag becomes false(0)(the loop will stop) if the mean square error of the present Y and the previous Y is less than 10^-6
        flag=0 if epsi<=1e-6 else 1
        #The Yvals array contains the array of all Y values for each iteration of the loop 
        Yvals.append(Yprednew[idx])
    #The bUpdate array, epoch count and the epsilon array with the mse for all the epochs is returned
    return np.array(bUpdate), epochs, eps_train, eps_test

#this function is used to generate the population set
def populationGenerator(x:float)->float:
    #A random value is generated from a normal distribution with mean=0 and standard deviation=5
    r=np.random.normal(0,5)
    #Generating the output value using the population generating equation => y=2x-3+N(0,5)
    y=2*(x)-3+r
    return y

#make sure the error is mse
#1000 pts for x, 80-20 train-test x->(-5,5)

def coeff(X:list[float], Y:list[float], n:int)->list[list[float]]:
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

#a main function is defined
def main():
    #An array of 1000 points is initialized for X values
    X_init=np.array(np.linspace(-5, 5, 1000))
    #the population is generated using the populationGenerator function for the whole input feature array X and returns an array of the population
    Y_init=[populationGenerator(X_init[i]) for i in range(len(X_init))]

    XYtup=[]
    for i in range(len(X_init)): 
        #creating a list of tuples => [(X,Y)]
        XYtup.append(tuple([X_init[i], Y_init[i]]))
    #inplace shuffling of the tuples
    np.random.shuffle(XYtup)

    #getting the training feature values and output
    X=np.array([XYtup[i][0] for i in range(len(X_init))])
    Y=np.array([XYtup[i][1] for i in range(len(X_init))])

        #The index for the first 80-20 split is computed
    split_idx=int(X.shape[0]*0.8)

    #A random sample is extracted off the XYtup, which is a list of tuples of the size of split_idx
    train_data = random.sample(XYtup, split_idx)
    #The data which is not in train_data but is in XYtup is utilized as the test data
    test_data = [i for i in XYtup if i not in train_data]
    #The X_test and Y_test value is extracted from the list of tuples test_data
    X_test, Y_test = zip(*test_data)
    #The X_test and Y_test value is extracted from the list of tuples test_data
    X_train, Y_train = zip(*train_data) 
    #The learning rate is initialized
    learning_rate=0.001

    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    X_test=np.array(X_test)
    Y_test=np.array(Y_test)

    #A random number from the normal distribution with mean 0 and std 1 is selected and initialized as b0 and b1
    b0=np.random.normal(0,1)
    b1=np.random.normal(0,1)
    beta=[b0, b1]#b0 and b1 is put into an array
    #The curried function is called with the randomly initialized values of beta 
    Y_pred=linearPolyModel(beta)
    #Here the predicted value for the input X is returned as Ypred
    Ypred=Y_pred(X_train)
    #We call the gradient descent function with the predicted value as Y_pred, 0.001 learning rate, the epochs for the gradient descent is also returned
    idx=np.random.randint(0, len(X_train))
    betaUpdate, epochs, eps_train, eps_test=gradDescent(X_train, Y_train, X_test, Y_test, Ypred[idx], beta, learning_rate)
    #Printing the epochs
    print(f"The total number of epochs that took to converge to the optimal β0 and β1 values from the stochastic gradient descent is : {epochs//len(X_train)}")
    #the optimal beta values are printed
    print(f"The β0 and β1 values calculated for training data using the Stochastic Gradient descent algorithm are : {betaUpdate[-1][0]}, {betaUpdate[-1][1]}")
    #The curried function is called with the beta values calculated through gradient descent
    Y_graddesc=linearPolyModel(betaUpdate[-1])
    #The output values for X with the betas from gradient descent is calculated
    Ygrad=Y_graddesc(X_train)
    #The actual beta values are stored in the betaActual array
    betaActual=coeff(X_train, Y_train, 1)
    #The curried function is called with the beta values calculated using closed form solution(matrix method)
    Y_act=linearPolyModel(betaActual)
    #The output values for X with betas from the closed form solution is calculated
    Yact=Y_act(X_train)
    #The mean squared error between the outputs generated by the gradient descent algorithm and the closed form solution is computed and printed
    eps_mse=meanSquaredError(Ygrad, Yact)
    eps_train.append(eps_mse)
    print(f"The ∈ value(Mean Square Error) after the obtaining β0 and β1 values from stochastic gradient descent is for training : {eps_train[-2]}")
    print(f"The ∈ value(Mean Square Error) after the obtaining β0 and β1 values from stochastic gradient descent is for testing : {eps_test[-1]}")
    #The function to plot the beta values vs epoch is called
    eps_train.pop()
    print(eps_train[-1]<eps_test[-1])
    #A plot for epoch vs mse for both training and testing datas is plotted
    plt.title("The Epoch vs Mean Square Error plot for Stochastic gradient descent")
    plt.plot(list(range(100, epochs+1)), eps_train[100:], c="r", label="MSE change with respect to epochs for training data(Bias)")
    plt.plot(list(range(100, epochs+1)), eps_test[100:], c="b", label="MSE change with respect to epochs for testing data(Variance)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean square error for training data")
    plt.figtext(0.5, 0.01, "This plot shows that the Mean square error for the model decreases as the epochs increases for the stochastic gradient descent algorithm", wrap=True, horizontalalignment='center', fontsize=12) # plot description
    plt.legend(loc="upper left")
    plt.show()

if __name__=="__main__":
    main()