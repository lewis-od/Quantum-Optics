import unittest
import numpy as np
from quoptics_old.states import Coherent, Cat

class TestCat(unittest.TestCase):

    def setUp(self):
        # Things tested here are implemented by the State base class, so
        # only need to test one of the State subclasses
        self.state = Cat(1j)

    def testData(self):
        alpha = Coherent(1j)
        minus = Coherent(-1j)
        expected = alpha.data + minus.data
        expected *= 1.0/np.sqrt(2*(1+np.exp(-2*np.abs(1j)**2)))
        self.assertTrue(np.all(expected == self.state.data))

    def testTheta(self):
        alpha = Coherent(1j)
        minus = Coherent(-1j)
        self.state.theta = np.pi
        expected = alpha.data + np.exp(1j*np.pi)*minus.data
        expected *= 1.0/np.sqrt(2*(1+np.cos(np.pi)*np.exp(-2*np.abs(1j)**2)))
        self.assertTrue(np.all(expected == self.state.data))
