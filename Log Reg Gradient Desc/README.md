# Classification using a Linear Regression

In this program, we convert the binary classified labels into 0s and 1s for the two labels. With this we find the closed form solution for each of the data point and return β0 and β1. With this we find the predicted value for the given X values, this returs a list of floats with a predicted value and we plot it. The decision boundary will be the normal to the regression line, and the points lower than the threshold limit will be a label and the points higher than that will be of another label. Also we're applying gradient descent on the sigmoid function and plotting it, while also trying to make the parameter a learnable from this sigmoid equation.

$$ {\text {S(x;a)}} = \frac{1}{1+e^-ax} $$

Try finding the gradient for the loss function of the sigmoid function

$$ {\text {L}} = \sum_{i} \frac{({\text {$y_{i}$ - $\hat{y_{i}}$})}^2} {N} $$

$$ where, y_{i} = \text{Actual output values} $$

$$ \hat{y_{i}}={\text{Predicted output/Estimated output}} $$

$$ N= \text{Total number of samples} $$ 
