import numpy as np

#Initializing a curried function linearPolyModel with a beta array as a parameter
def linearPolyModel(beta:list[float]):
    #A function _impl is defined inside linearPolyModel to curry the function with x as input
    def _impl(x:float)->float:
        #calcs the b0+b1*x value for the given x
        return beta[0] + (beta[1]*x)
    #returns the value of the function _impl
    return _impl

def meanSquaredError(y:int, y_pred:int)->int:
    #Here we calculate the mean square error with the formula sum((y-y_pred)**2)/size(y)
    return np.mean((y-y_pred)**2)


def gradDescent(X:list[float], Y:list[float], Y_pred:list[float], beta:list[float], learning_rate:int)->list[list[float]]:
    #an array to add a tuple with b0 and b1 value for each epoch, it is initialized with the initial b0 and b1 values
    bUpdate=[[beta[0], beta[1]]]
    #Epoch counter initlialized to 0
    eps=0
    #The Yvals array is initialised with the first predicted value of Y with random b0 and b1 generated, that is closer to 0
    Yvals=[Y_pred]
    #The flag is set to 1
    flag=1
    #This loop executes until the flag value becomes false
    while(flag):
        #The b0 and b1 value of the last inserted value of the element is used as b0 and b1(changes for each epoch)
        b0,b1=bUpdate[-1][0],bUpdate[-1][1] 
        #partial derivative of error func wrt b0 is computed(db/db0=(2y-y_pred)(-1))
        dedb0=np.mean(-2*(Y - Yvals[-1])) 
        #partial derivative of error func wrt b1 is computed(db/db0=(2y-y_pred)(-x))
        dedb1=np.mean(-2*((Y - Yvals[-1])* X))
        #The value of b0 is updated for each epoch, each value of b0 and b1
        #b0 old= b0 new+ learning_rate*(de/db0)
        b0-=(learning_rate*dedb0)
        #b1 old= b1 new+ learning_rate*(de/db1)
        b1-=(learning_rate*dedb1)
        # This array contains the value of b0 and b1 for each iteration
        betArr=[b0, b1] 
        #The curried function is called with the beta array
        Y_pred_new=linearPolyModel(betArr)
        #The curryied function is used to generate the value of Yprednew for each value of X 
        Yprednew=Y_pred_new(X)
        #The updated b0 and b1 value is appened into bUpdate as an array -> [b0, b1]
        bUpdate.append(betArr) 
        #The epochs is counted for each iteration
        eps+=1
        #The flag becomes false(0)(the loop will stop) if the mean square error of the present Y and the previous Y is less than 10^-6
        flag=0 if meanSquaredError(Yprednew, Yvals[-1])<=10**-6 else 1
        #The Yvals array contains the array of all Y values for each iteration of the loop 
        Yvals.append(Yprednew)
    print(eps) 
    #The bUpdate array is returned after completing all the epochs
    return np.array(bUpdate)

#this function is used to generate the population set
def populationGenerator(x:int)->int:
    #A random value is generated from a normal distribution with mean=0 and standard deviation=5
    r=np.random.normal(0,5)
    #Generating the output value using the population generating equation => y=2x-3+N(0,5)
    y=2*(x)-3+r
    return y

#make sure the error is mse
#1000 pts for x, 80-20 train-test x->(-5,5)

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

    #The learning rate is initialized
    learning_rate=0.001

    #A random number from the normal distribution with mean 0 and std 1 is selected and initialized as b0 and b1
    b0=np.random.normal(0,1)
    b1=np.random.normal(0,1)
    beta=[b0, b1]#b0 and b1 is put into an array

    #The curried function is called with the randomly initialized values of beta 
    Y_pred=linearPolyModel(beta)
    #Here the predicted value for the input X is returned as Ypred
    Ypred=Y_pred(X)
    #We call the gradient descent function with the predicted value as Y_pred, 0.001 learning rate and 1000 epochs
    betaUpdate=gradDescent(X, Y, Ypred, beta, learning_rate)
    #the optimal beta values are printed
    print(betaUpdate[-1])
    #The curried function is called with the beta values calculated through gradient descent
    Y_graddesc=linearPolyModel(betaUpdate[-1])
    #The output values for X with the betas from gradient descent is calculated
    Ygrad=Y_graddesc(X)
    #The actual beta values are stored in the betaActual array
    betaActual=coeff(X, Y, 1)
    #The curried function is called with the beta values calculated using closed form solution(matrix method)
    Y_act=linearPolyModel(betaActual)
    #The output values for X with betas from the closed form solution is calculated
    Yact=Y_act(X)
    #The mean squared error between the outputs generated by the gradient descent algorithm and the closed form solution is printed
    print(meanSquaredError(Ygrad, Yact))

if __name__=="__main__":
    main()