import unittest
import numpy as np
from quoptics_old.states import Fock

class TestFock(unittest.TestCase):

    def testChangeN(self):
        state = Fock(0)
        with self.assertRaises(ValueError):
            state.n = 0.5

    def testOperator(self):
        state = Fock(17)
        old = state.data
        state.analytic = False
        self.assertAlmostEqual(max(old), max(state.data))
        self.assertEqual(np.argmax(old), np.argmax(state.data))
