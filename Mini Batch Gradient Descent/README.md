# Mini Batch Gradient Descent Simulation

This program simulates the working of Gradient Descent. This outputs a graph that displays the relationship between the epochs and error for each epoch, each of the epoch will generate a new epoch using the equation

$$β_{new} = β_{old} + \eta\frac{\partial \epsilon}{\partial \beta}$$

Here, the betas are calculated for each batch, (where batch_size=50) for each epoch until the beta values converge.