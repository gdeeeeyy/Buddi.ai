# Monte Carlo Simulation for $$\pi$$ estimation

This is a Monte Carlo simulation that is used to estimate the value of pi by throwing darts onto an unit square which contains an unit circle, so the probability of the darts falling onto the circle will be pi/4 and the value of pi will be 4 times the probability of darts falling into the unit circle. This graph shows the estimated pi value for total number of darts thrown onto the unit square.

P(darts falling into unit circle) = $$\frac{\text {Number of darts inside the unit circle}}{\text {Total number of darts}}$$ = area(unit circle)/area(square)=$$\frac{ \pi } {\text {4}}$$

Therefore, $$\pi_{estimate}$$ = 4\*P(darts falling into the unit circle)

$$\pi_{estimate}$$ = 4\*($$\frac{\text {Number of darts inside the unit circle}}{\text {Total number of darts}}$$)

This $$\pi_{estimate}$$ is the estimated value of $$\pi$$, which can change for different values of the total number of darts.

The output for this code is:

For Uniform distribution(The darts are thrown at random co-ordinates chosen from the uniform distribution):
![Screenshot from 2024-05-13 12-45-23](https://github.com/gdeeeeyy/Buddi.ai/assets/73658032/192571c5-bfa8-4d7f-a95e-cb2c2609e16f)
![Screenshot from 2024-05-13 12-45-14](https://github.com/gdeeeeyy/Buddi.ai/assets/73658032/8e726ba4-51ed-4139-9edb-801c47674679)

For Uniform distribution(The darts are thrown at random co-ordinates chosen from the normal distribution with mean 0 and standard deviation 3):
![Screenshot from 2024-05-13 12-48-26](https://github.com/gdeeeeyy/Buddi.ai/assets/73658032/2faf376d-ae74-4e94-a288-b063016d42ec)
![Screenshot from 2024-05-13 12-48-20](https://github.com/gdeeeeyy/Buddi.ai/assets/73658032/515efe21-8238-4919-a5bd-060f7d60e6e9)
