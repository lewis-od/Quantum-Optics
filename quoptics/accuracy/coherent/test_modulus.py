import numpy as np
import matplotlib.pyplot as plt
from ... import conf
from ... import states

conf.T = 50 # Truncate at 50
alphas = np.linspace(0, 10, 100)
deltas = np.zeros(100) # Array for storing differences of calculates states
analytic = states.Coherent(0, analytic=True)
matrix = states.Coherent(0, analytic=False)
for i in range(len(alphas)):
    analytic.alpha = alphas[i]
    matrix.alpha = alphas[i]
    # Calculate the difference in the 2 methods of calculation
    delta = analytic.data - matrix.data
    # Store the norm of the difference of the vectors
    deltas[i] = np.linalg.norm(delta)

# Plot the results
plt.figure()
plt.plot(alphas, deltas)
plt.title('Difference between analytical and operator approximations')
plt.xlabel('$|\\alpha$|')
plt.ylabel('Difference')
plt.show()
