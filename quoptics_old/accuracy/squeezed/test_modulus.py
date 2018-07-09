import numpy as np
import matplotlib.pyplot as plt
from ... import conf
from ... import states

conf.T = 60 # Truncate at 50
zs = np.linspace(0, 10, 100)
deltas = np.zeros(100) # Array for storing differences of calculates states
analytic = states.Squeezed(0, analytic=True)
matrix = states.Squeezed(0, analytic=False)
for i in range(len(zs)):
    analytic.z = zs[i]
    matrix.z = zs[i]
    # Calculate the difference in the 2 methods of calculation
    delta = analytic.data - matrix.data
    # Store the norm of the difference of the vectors
    deltas[i] = np.linalg.norm(delta)

# Plot the results
plt.figure()
plt.plot(zs, deltas)
plt.title('Difference between analytical and operator approximations')
plt.xlabel('z')
plt.ylabel('Difference')
plt.show()
