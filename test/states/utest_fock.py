import unittest
from quoptics.states import Fock

class TestFock(unittest.TestCase):

    def testChangeN(self):
        state = Fock(0)
        with self.assertRaises(ValueError):
            state.n = 0.5
        
