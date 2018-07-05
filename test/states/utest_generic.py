import unittest
import numpy as np
from quoptics.states import Generic, Coherent

class TestGenericState(unittest.TestCase):

    def testTruncation(self):
        state = Generic()
        self.assertEqual(state.T, 0)
        state.data = np.array([1, 2, 3, 4, 5])
        self.assertEqual(state.T, 5)

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
