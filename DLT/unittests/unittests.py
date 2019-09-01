import unittest

import numpy as np

import DLT

class TestDLT(unittest.TestCase):
    def test_computeCentroid(self):
        points = np.array([[1,2], [3,4]])
        centroid = DLT.computeCentroid(points)
        self.assertTrue(np.array_equal(centroid, [2., 3.]))

    def test_computeTransformation(self):
        points = np.array([[1, 1, 1], [1, -1, 1], [-1, -1 , 1], [-1, 1, 1]])
        H = DLT.computeDLTTransformation(points, points)
        
        expectedTransformation = np.identity(3)
        self.assertTrue(np.allclose(H, expectedTransformation))

if __name__ == "__main__":
    unittest.main()

