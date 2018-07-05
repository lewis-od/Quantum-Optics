import unittest
import numpy as np
from quoptics.states import Generic, Fock, Coherent, Squeezed

class TestGenericState(unittest.TestCase):

    def testTruncation(self):
        state = Generic()
        self.assertEqual(state.T, 0)
        state.data = np.array([1, 2, 3, 4, 5])
        self.assertEqual(state.T, 5)

    def testCreateFock(self):
        n = 5
        gen = Generic(Fock(n).data)
        fock = Fock.from_generic(gen)
        self.assertTrue(np.all(gen.data == fock.data))
        self.assertEqual(fock.n, n)

        gen = Generic(Fock(n, analytic=False).data)
        fock = Fock.from_generic(gen)
        self.assertTrue(np.all(gen.data == fock.data))
        self.assertEqual(fock.n, n)

    def testCreateCoherent(self):
        alpha = 1+1j
        gen = Generic(Coherent(alpha).data)
        coherent = Coherent.from_generic(gen)
        self.assertEqual(coherent.alpha, alpha)
        self.assertTrue(np.all(coherent.data == gen.data))

        gen.data = Coherent(alpha, analytic=False).data
        coherent = Coherent.from_generic(gen)
        self.assertAlmostEqual(alpha, coherent.alpha)
        self.assertTrue(np.all(coherent.data == gen.data))

    def testCreateSqueezed(self):
        z = 1+1j
        gen = Generic(Squeezed(z).data)
        squeezed = Squeezed.from_generic(gen)
        self.assertAlmostEqual(squeezed.z, z)
        self.assertTrue(np.all(gen.data == squeezed.data))

        gen = Generic(Squeezed(z, T=200, analytic=False).data)
        squeezed = Squeezed.from_generic(gen)
        self.assertAlmostEqual(z, squeezed.z)
        self.assertTrue(np.all(gen.data == squeezed.data))
