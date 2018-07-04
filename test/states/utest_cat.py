import unittest
import numpy as np
from quoptics.states import Coherent, Cat

class TestCat(unittest.TestCase):

    def setUp(self):
        # Things tested here are implemented by the State base class, so
        # only need to test one of the State subclasses
        self.state = Cat(1j)

    def testData(self):
        alpha = Coherent(1j)
        minus = Coherent(-1j)
        expected = alpha.data + minus.data
        expected *= 1.0/np.sqrt(2)
        self.assertTrue(np.all(expected == self.state.data))

    def testChangeSign(self):
        with self.assertRaises(ValueError):
            self.state.sign = 'a'
        old = list(self.state.data)
        self.state.sign = '-'
        self.assertNotEqual(old, list(self.state.data))
