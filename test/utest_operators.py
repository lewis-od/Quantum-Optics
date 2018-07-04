import unittest
import numpy as np
import quoptics.operators as ops
from quoptics.states import Fock, Coherent, Squeezed

class TestOperators(unittest.TestCase):

    def testCreation(self):
        a = ops.annihilation(50)
        c = ops.creation(50)
        self.assertTrue(np.all(c == a.T))

    def testNumber(self):
        a = ops.annihilation(50)
        c = ops.creation(50)
        n = ops.number(50)
        self.assertTrue(np.all((c @ a) == n))

    def testDisplacement(self):
        D = ops.displacement(1j, 50)
        vacuum = Fock(0).data
        state = Coherent(1j, analytic=False)
        displaced = D @ vacuum
        self.assertTrue(np.all(displaced == state.data))

    def testSqueesing(self):
        S = ops.squeezing(1j, 50)
        vacuum = Fock(0).data
        state = Squeezed(1j, analytic=False)
        squeezed = S @ vacuum
        self.assertTrue(np.all(squeezed == state.data))
