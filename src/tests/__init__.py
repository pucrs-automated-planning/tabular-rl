import rl.agent
import unittest
from tests.tabular_test import TestTabularAgents


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestTabularAgents))
    return suite
