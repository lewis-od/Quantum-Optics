import numpy as np
import matplotlib.pyplot as plt
from ... import states

T = 60 # Truncate at 50
thetas = np.linspace(-np.pi, np.pi, 100)
deltas = np.zeros(100) # Array for storing differences of calculates states
for i in range(len(thetas)):
    z = np.exp(1j*thetas[i])
    # Calculate the difference in the 2 methods of calculation
    delta = states.squeezed1(z, T=T) - states.squeezed2(z, T=T)
    # Store the norm of the difference of the vectors
    deltas[i] = np.linalg.norm(delta)

# Plot the results
plt.figure()
plt.plot(thetas, deltas)
plt.title('Difference between analytical and operator approximations')
plt.xlabel('Arg(z)')
plt.ylabel('Difference')
plt.show()
