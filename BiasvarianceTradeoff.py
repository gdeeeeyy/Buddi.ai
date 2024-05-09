import numpy as np
import matplotlib.pyplot as plt

def populationGenerator(x:int)->int:
    r=np.random.normal(0,3)
    y=(2*(x**4))-(3*(x**3))+(7*(x**2))-(23*(x))+8+r
    return y

X=np.array(np.linspace(-5,5,100))
n=len(X)
Y=populationGenerator(X)

split_idx_train=int(X.shape[0]*0.7)
split_idx_valid=split_idx_train+int(X.shape[0]*0.1)


X1_train=np.array(X[:split_idx_train])
X0_train=np.array(X1_train**0)
X2_train=np.array(X1_train**2)
X3_train=np.array(X1_train**3)
X4_train=np.array(X1_train**4)
Y_train=np.array(Y[:split_idx_train])


X1_valid=np.array(X[split_idx_train:])
X0_valid=np.array([X1_valid**0])
X2_valid=np.array([X1_valid**2])
X3_valid=np.array([X1_valid**3])
X4_valid=np.array([X1_valid**4])
Y_valid=np.array([Y[split_idx_train:]])

Xtrans=np.array([X0_train, X1_train, X2_train, X3_train, X4_train])
X_y=np.transpose(Xtrans)
XInv=np.linalg.inv(np.matmul(Xtrans, X_y))
calc1=np.matmul(XInv, Xtrans)
beta=np.matmul(calc1, Y_train)

def linearModel(X1:list[float], beta:list[float])->list[float]:
    return beta[0]+(beta[1]*X1)

def quadraticModel(X1:list[float], X2:list[float], beta:list[float])->list[float]:
    return beta[0]+(beta[1]*X1)+(beta[2]*X2)

def cubicModel(X1:list[float], X2:list[float], X3:list[float], beta:list[float])->list[float]:
    return beta[0]+(beta[1]*X1)+(beta[2]*X2)+(beta[3]*X3)

def quarternaryModel(X1:list[float], X2:list[float], X3:list[float], X4:list[float], beta:list[float])->list[float]:
    return beta[0]+(beta[1]*X1)+(beta[2]*X2)+(beta[3]*X3)+((beta[4]*X4))

def lagrangesPolynomial(X:list[float], Y:list[float], xi:int, n:int)->int:
    res=0.0
    for i in range(n):
        t=Y[i]
        for j in range(n):
            if(j!=i):
                t=t*(xi-X[j])/(X[i]-X[j])
        res+=t
    return res

def models(X:list[float], Y:list[float], beta:list[float])->list[list[float]]:
    X1=X
    X2=np.array(X1**2)
    X3=np.array(X1**3)
    X4=np.array(X1**4)
    lin_model=linearModel(X1, beta)
    quad_model=quadraticModel(X1, X2, beta)
    cub_model=cubicModel(X1, X2, X3, beta)
    quart_model=quarternaryModel(X1, X2, X3, X4, beta)
    lagranges_model=[lagrangesPolynomial(X1,Y,X1[i],len(X1)) for i in range(len(X1))]
    mod=[lin_model, quad_model, cub_model, quart_model, lagranges_model]
    return mod

models=models(X, Y, beta)


def plot(X:list[float], Y:list[float], models:list[float])->int:
    txt="Plot that shows the performance of the Linear, quadratic, cubic, biquadratic and lagrange's polynomial model for the given inputs and outputs"
    plt.scatter(X, Y, marker=".", label="Actual Values")
    plt.title("Model Performance Estimation(Linear, Quadratic, Cubic, Quarternary, Lagrange's polynomial)")
    plt.plot(X, models[0], label="Linear polynomial model")
    plt.plot(X, models[1], label="Quadratic polynomial model")
    plt.plot(X, models[2], label="Cubic polynomial model")
    plt.plot(X, models[3], label="Quarternary polynomial model")
    plt.plot(X, models[4], label="Lagrange's polynomial model", c="r")
    plt.xlabel("Feature values")
    plt.ylabel("Output values")
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=8)
    plt.legend()
    plt.show()
    plt.close()
    return 1
plot(X,Y,models)

def train(X1_train:list[float], X2_train:list[float], X3_train:list[float], X4_train:list[float], beta:list[float])->list[float]:
    lin_model_train=linearModel(X1_train, beta)
    quad_model_train=quadraticModel(X1_train, X2_train, beta)
    cub_model_train=cubicModel(X1_train, X2_train, X3_train, beta)
    quart_model_train=quarternaryModel(X1_train, X2_train, X3_train, X4_train, beta)
    lagranges_model_train=[lagrangesPolynomial(X1_train,Y_train,X1_train[i],len(X1_train)) for i in range(len(X1_train))]

    eps_lin_train = np.sum(np.abs(Y_train - lin_model_train) / len(X1_train))
    eps_quad_train = np.sum(np.abs(Y_train - quad_model_train) / len(X1_train))
    eps_cube_train = np.sum(np.abs(Y_train - cub_model_train) / len(X1_train))
    eps_quart_train = np.sum(np.abs(Y_train - quart_model_train) / len(X1_train))
    eps_lag_train = np.sum(np.abs(Y_train - lagranges_model_train) / len(X1_train))
    op=[eps_lin_train, eps_quad_train, eps_cube_train, eps_quart_train, eps_lag_train]
    return op

eps_bias=train(X1_train, X2_train, X3_train, X4_train, beta)

def valid(X1_valid:list[float], X2_valid:list[float], X3_valid:list[float], X4_valid:list[float], beta:list[float])->list[float]:
    lin_model_valid=linearModel(X1_valid, beta)
    quad_model_valid=quadraticModel(X1_valid, X2_valid, beta)
    cub_model_valid=cubicModel(X1_valid, X2_valid, X3_valid, beta)
    quart_model_valid=quarternaryModel(X1_valid, X2_valid, X3_valid, X4_valid, beta)
    # lagranges_model_valid=[lagrangesPolynomial(X1_valid,Y_valid,X1_valid[i],70) for i in range(len(X1_train))]

    eps_lin_valid = np.sum(np.abs(Y_valid - lin_model_valid) / len(X1_valid))
    eps_quad_valid = np.sum(np.abs(Y_valid - quad_model_valid) / len(X1_valid))
    eps_cube_valid = np.sum(np.abs(Y_valid - cub_model_valid) / len(X1_valid))
    eps_quart_valid = np.sum(np.abs(Y_valid - quart_model_valid) / len(X1_valid))
    eps=[eps_lin_valid, eps_quad_valid, eps_cube_valid, eps_quart_valid]
    # eps_lag_model_valid=np.sum(np.abs(Y_valid-lagranges_model_valid))
    return eps #eps_lag_model_valid

eps_variance=valid(X1_valid, X2_valid, X3_valid, X4_valid, beta)

print(eps_bias)
print(eps_variance)


txt="This plot presents the Bias-variance trade off for the Linear, Quadratic, Cubic, Quarternary models and a Lagrange's Polynomial"
x=[1,2,3,4,70]
plt.plot(x, eps_bias, c="r", label="Bias", marker=".")
plt.title("Bias-Variance Trade off")
plt.xlabel("Model Complexity")
plt.ylabel("Error estimate")
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=8)
plt.plot(x[:4], eps_variance, c="b", label="Variance", marker=".")
plt.legend()
plt.show()