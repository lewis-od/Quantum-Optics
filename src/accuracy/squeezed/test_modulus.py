import numpy as np
import matplotlib.pyplot as plt
from ... import states

T = 60 # Truncate at 50
zs = np.linspace(0, 10, 100)
deltas = np.zeros(100) # Array for storing differences of calculates states
for i in range(len(zs)):
    z = zs[i]
    # Calculate the difference in the 2 methods of calculation
    delta = states.squeezed1(z, T=T) - states.squeezed2(z, T=T)
    # Store the norm of the difference of the vectors
    deltas[i] = np.linalg.norm(delta)

# Plot the results
plt.figure()
plt.plot(zs, deltas)
plt.title('Difference between analytical and operator approximations')
plt.xlabel('z')
plt.ylabel('Difference')
plt.show()
