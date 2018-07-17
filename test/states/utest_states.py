import unittest
import numpy as np
from quoptics_old.states import Coherent, Fock

class TestStates(unittest.TestCase):

    def setUp(self):
        # Things tested here are implemented by the State base class, so
        # only need to test one of the State subclasses
        self.state = Coherent(1j)

    def testDataReadonly(self):
        with self.assertRaises(AttributeError):
            self.state.data = []

    def testChangeParameter(self):
        old = list(self.state.data)
        self.state.alpha = 1.0
        self.assertNotEqual(old, list(self.state.data))

    def testChangeTruncation(self):
        self.state.T = 100
        self.assertEqual(len(self.state.data), 100)

    def testChangeAnalytic(self):
        old = self.state.data
        self.state.analytic = False
        for n in range(len(old)):
            self.assertAlmostEqual(old[n], self.state.data[n])

    def testInnerProduct(self):
        psi = Fock(4)
        c = self.state.inner_prod(psi)
        self.assertEqual(self.state.data[4], c)

    def testNorm(self):
        norm = self.state.norm()
        inner = self.state.inner_prod(self.state)
        self.assertEqual(norm, inner)
        self.assertTrue(norm <= 1)
