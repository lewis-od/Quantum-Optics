import numpy as np
import matplotlib.pyplot as plt
from ... import states

T = 50 # Truncate at 50
alphas = np.linspace(0, 10, 100)
deltas = np.zeros(100) # Array for storing differences of calculates states
for i in range(len(alphas)):
    alpha = alphas[i]
    # Calculate the difference in the 2 methods of calculation
    delta = states.coherent(alpha, T=T) - states.coherent2(alpha, T=T)
    # Store the norm of the difference of the vectors
    deltas[i] = np.linalg.norm(delta)

# Plot the results
plt.figure()
plt.plot(alphas, deltas)
plt.title('Difference between analytical and operator approximations')
plt.xlabel('$|\\alpha$|')
plt.ylabel('Difference')
plt.show()
