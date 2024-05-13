# Monte Carlo Simulation for $$\pi$$ estimation

This is a Monte Carlo simulation that is used to estimate the value of pi by throwing darts onto an unit square which contains an unit circle, so the probability of the darts falling onto the circle will be pi/4 and the value of pi will be 4 times the probability of darts falling into the unit circle. This graph shows the estimated pi value for total number of darts thrown onto the unit square.

P(darts falling into unit circle) = $$\frac{\text {Number of darts inside the unit circle}}{\text {Total number of darts}}$$ = area(unit circle)/area(square)=$$\frac{ \pi } {\text {4}}$$

Therefore, $$\pi_{estimate}$$ = 4\*P(darts falling into the unit circle)

$$\pi_{estimate}$$ = 4\*($$\frac{\text {Number of darts inside the unit circle}}{\text {Total number of darts}}$$)

This $$\pi_{estimate}$$ is the estimated value of $$\pi$$, which can change for different values of the total number of darts.

The output for this code is:

For Uniform distribution(The darts are thrown at random co-ordinates chosen from the uniform distribution):

![Alt text]("/home/krishna/Pictures/Screenshots/Screenshot from 2024-05-13 12-45-23.png")
![Alt text]("/home/krishna/Pictures/Screenshots/Screenshot from 2024-05-13 12-45-14.png")

For Uniform distribution(The darts are thrown at random co-ordinates chosen from the normal distribution with mean 0 and standard deviation 3):

![Alt text]("/home/krishna/Pictures/Screenshots/Screenshot from 2024-05-13 12-48-26.png")
![Alt text]("/home/krishna/Pictures/Screenshots/Screenshot from 2024-05-13 12-48-20.png")
