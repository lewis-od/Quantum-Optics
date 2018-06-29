import numpy as np
import matplotlib.pyplot as plt
from ... import conf
from ... import states

conf.T = 50
thetas = np.linspace(-np.pi, np.pi, 100)
alphas = np.array([np.exp(1j*theta) for theta in thetas])
deltas = np.zeros(100)
analytic = states.Coherent(0, analytic=True)
matrix = states.Coherent(0, analytic=False)
for i in range(len(alphas)):
    analytic.alpha = alphas[i]
    matrix.alpha = alphas[i]
    delta = analytic.data - matrix.data
    deltas[i] = np.linalg.norm(delta)

plt.figure()
plt.plot(thetas, deltas)
plt.title('Difference between analytical and operator approximations')
plt.xlabel('Arg($\\alpha$)')
plt.ylabel('Difference')
plt.show()
