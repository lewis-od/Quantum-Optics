import numpy as np
import matplotlib.pyplot as plt
from ... import states

T = 50
thetas = np.linspace(-np.pi, np.pi, 100)
alphas = np.array([np.exp(1j*theta) for theta in thetas])
deltas = np.zeros(100)
for i in range(len(alphas)):
    alpha = alphas[i]
    delta = states.coherent(alpha, T=T) - states.coherent2(alpha, T=T)
    deltas[i] = np.linalg.norm(delta)

plt.figure()
plt.plot(thetas, deltas)
plt.title('Difference between analytical and operator approximations')
plt.xlabel('Arg($\\alpha$)')
plt.ylabel('Difference')
plt.show()
