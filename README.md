# Buddi.ai assignments

## Grid search

This is a grid search code for getting the optimal b1 and b2 value with the least total error from the quadratic linear equation with 0.01 step range. An error surface plot is also plotted to output a graph to represent the optimal minima from the given range of b1 and b2 with the given inputs and outputs. (This error surface plot is only used to visually represent the global minima and not to find it, as we have already estimated the minima.)

**To add: Multiprocessing**

## Markov Chain simulation to estimate value of pi

This is a Monte Carlo simulation that is used to estimate the value of pi by throwing darts onto an unit square which contains an unit circle, so the probability of the darts falling onto the circle will be pi/4 and the value of pi will be 4 times the probability of darts falling into the unit circle. This graph shows the estimated pi value for total number of darts thrown onto the unit square.

P(darts falling into unit circle)= number of darts inside the unit circle/Total number of darts=area(unit circle)/area(square)=pi\*((a**2)/2)/a**2=pi/4

Therefore, pi=P(darts falling into the unit circle)*4
pi_cap=(number of darts inside the unit circle/Total number of darts)*4

This pi_cap is the estimated value of pi, which can change for different values of the total number of darts.

**To add: Smoothen the curve(This one was done using google sheets) and Multiprocessing**

### Using uniform random samples

The random values, (i.e) the random points that the darts land on the unit square is decided by taking samples from an uniform distribution, this would give us randomized values, which can fall on any part of the unit square, this helps us to estimate the value of pi which can make pi_cap converge faster.

### Using random samples from normal distribution

The random values, (i.e) the random points that the darts land on the unit square is decided by taking samples from a normal distribution, this would give us randomized values, which can fall in the centre of the unit square, that is the unit circle, if the standard deviation of the normal distribution is 1, if the standard deviation is increased, the variance increases and the points can fall on any points on the unit circle, which can help us to converge the value of pi_cap.

## Random Sampler

This program is used to construct a random sampler using a distribution with varying Probability mass functions, this may make the random sampling more weighted for a given feature, but we can calculate the cummulative mass function for each of the features to make the sampling more random and not weighted.

## Normal curve

A gaussian is plotted for the given normal function, two graphs are plotted where one has 3 different plots with the same mean and different standard
deviations and the other one also has 3 different plots with same standard deviations and different means, and we can visually notice the difference
between the normal curves.

## Lagrange's Polynomial

The Lagrange's polynomial is used to estimate the given points perfectly, i.e it creates a function so that all the given data points are covered in the graph. Note: Lagrange's polynomial can't be used to create a model that can be deployed as it doesn't work well on unseen data.

## Bias Variance Trade Off

The bias variance trade off graph represents the relationship of the training error i,e the bias and the testing error i,e the variance with the complexity of the model(the degree of the model). We can see that after some point, eventhough the bias is decreasing, the variance increases with increasing complexity of the model

## Gradient Descent

The gradient descent algorithm is applied for the 1000 feature values between -5, 5, we find the closed form solution for the given population generator function, the gradient descent algorithm is used to find the optimal value of b0 and b1, where the mean squared error is 10^-6. Finally the difference between the output from the closed form solution and the output from the gradient descent algorithm
