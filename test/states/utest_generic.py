import unittest
import numpy as np
from quoptics.states import Generic

class TestGenericState(unittest.TestCase):

    def testTruncation(self):
        state = Generic()
        self.assertEqual(state.T, 0)
        state.data = np.array([1, 2, 3, 4, 5])
        self.assertEqual(state.T, 5)
