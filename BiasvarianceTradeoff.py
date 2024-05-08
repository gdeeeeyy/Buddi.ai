import numpy as np
import matplotlib.pyplot as plt

def populationGenerator(x:int)->int:
    r=np.random.normal(0,3)
    y=(2*(x**4))-(3*(x**3))+(7*(x**2))-(23*(x))+8+r
    return y

X=np.array(np.linspace(-5,5,100))
n=len(X)

split_idx_train=int(X.shape[0]*0.7)
split_idx_valid=split_idx_train+int(X.shape[0]*0.1)


X1_train=X[:split_idx_train]
X0_train=np.array(X1_train**0)
X2_train=np.array(X1_train**2)
X3_train=np.array(X1_train**3)
X4_train=np.array(X1_train**4)
Y_train=np.array(populationGenerator(X1_train))


X1_valid=np.array(X[split_idx_train:split_idx_valid])
X0_valid=np.array([X1_valid**0])
X2_valid=np.array([X1_valid**2])
X3_valid=np.array([X1_valid**3])
X4_valid=np.array([X1_valid**4])
Y_valid=np.array([populationGenerator(X1_valid)])


X1_test=np.array(X[split_idx_valid:])
X0_test=np.array([X1_test**0])
X2_test=np.array([X1_test**2])
X3_test=np.array([X1_test**3])
X4_test=np.array([X1_test**4])
Y_test=np.array([populationGenerator(X1_test)])


Xtrans=np.array([X0_train, X1_train, X2_train, X3_train, X4_train])
X=np.transpose(Xtrans)
XInv=np.linalg.inv(np.matmul(Xtrans, X))
calc1=np.matmul(XInv, Xtrans)
beta=calc1=np.matmul(calc1, Y_train)

def linear_model(X1:list[float], beta:list[float])->list[float]:
    return beta[0]+(beta[1]*X1)

def quadratic_model(X1, X2, beta):
    return beta[0]+(beta[1]*X1)+(beta[2]*X2)

def cubic_model(X1, X2, X3, beta):
    return beta[0]+(beta[1]*X1)+(beta[2]*X2)+(beta[3]*X3)

def quarternary_model(X1, X2, X3, X4, beta):
    return beta[0]+(beta[1]*X1)+(beta[2]*X2)+(beta[3]*X3)+(+(beta[4]*X4))

def lagrangesPolynomial(X, Y, xi, n):
    res=0.0
    for i in range(n):
        t=Y[i]
        for j in range(n):
            if(j!=i):
                t=t*(xi-X[j])/(X[i]-X[j])
        res+=t
    return res

def train():
    lin_model_train=linear_model(X1_train, beta)
    quad_model_train=quadratic_model(X1_train, X2_train, beta)
    cub_model_train=cubic_model(X1_train, X2_train, X3_train, beta)
    quart_model_train=quarternary_model(X1_train, X2_train, X3_train, X4_train, beta)
    lagranges_model_train=[lagrangesPolynomial(X1_train,Y_train,X1_train[i],len(X1_train)) for i in range(len(X1_train))]

    eps_linear_model_train=np.sum(np.abs(Y_train-lin_model_train))/len(X1_train)
    eps_quad_model_train=np.sum(np.abs(Y_train-quad_model_train))/len(X1_train)
    eps_cub_model_train=np.sum(np.abs(Y_train-cub_model_train))/len(X1_train)
    eps_quart_model_train=np.sum(np.abs(Y_train-quart_model_train))/len(X1_train)
    eps_lag_model_train=np.sum(np.abs(Y_train-lagranges_model_train))/len(X1_train)
    return [eps_linear_model_train, eps_quad_model_train, eps_cub_model_train, eps_quart_model_train, eps_lag_model_train]

eps_bias=train()

def valid():
    lin_model_valid=linear_model(X1_valid, beta)
    quad_model_valid=quadratic_model(X1_valid, X2_valid, beta)
    cub_model_valid=cubic_model(X1_valid, X2_valid, X3_valid, beta)
    quart_model_valid=quarternary_model(X1_valid, X2_valid, X3_valid, X4_valid, beta)
    # lagranges_model_valid=[lagrangesPolynomial(X1_valid,Y_valid,X1_valid[i],70) for i in range(len(X1_train))]

    eps_linear_model_valid=np.sum(np.abs(Y_valid-lin_model_valid))/len(X1_valid)
    eps_quad_model_valid=np.sum(np.abs(Y_valid-quad_model_valid))/len(X1_valid)
    eps_cub_model_valid=np.sum(np.abs(Y_valid-cub_model_valid))/len(X1_valid)
    eps_quart_model_valid=np.sum(np.abs(Y_valid-quart_model_valid))/len(X1_valid)
    eps=[eps_linear_model_valid, eps_quad_model_valid, eps_cub_model_valid, eps_quart_model_valid]
    # eps_lag_model_valid=np.sum(np.abs(Y_valid-lagranges_model_valid))
    return [eps_linear_model_valid, eps_quad_model_valid, eps_cub_model_valid, eps_quart_model_valid] #eps_lag_model_valid

eps_variance=valid()

print(eps_bias)
print(eps_variance)

txt="This plot presents the Bias-variance trade off for the Linear, Quadratic, Cubic, Quarternary models and a Lagrange's Polynomial"
x=[1,2,3,4,70]
plt.plot(x, eps_bias, c="r", label="Bias")
plt.title("Bias-Variance Trade off")
plt.xlabel("Model Complexity")
plt.ylabel("Error estimate")
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=8)
plt.plot(x[:4], eps_variance, c="b", label="Variance")
plt.legend()
plt.show()