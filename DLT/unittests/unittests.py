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

    def test_computeTransformationForSamePoints(self):
        points = np.array([[1,1,1], [2,1,1], [1.5,2,1], [1.5,3,1], [3,2, 1]])
        H = DLT.computeDLTTransformation(points, points)
        expectedTransformation = np.identity(3)
        self.assertTrue(np.allclose(H, expectedTransformation))

    def test_computeTransformationAffine(self):
        points = np.array([[1,1,1], [2,1,1], [1.5,2,1], [1.5,3,1], [3,2, 1]])
        
        affineTransformation = np.identity(3)
        affineTransformation[0, 0] = 3
        affineTransformation[1, 1] = 5
        affineTransformation[:, 2] = [2, 5, 1]
        expectedPoints = [np.dot(affineTransformation, np.transpose(point)) for point in points]

        H = DLT.computeDLTTransformation(points, np.asarray(expectedPoints))

        measuredPoints = [np.dot(H, np.transpose(point)) for point in points]
        self.assertTrue(np.allclose(expectedPoints, measuredPoints))

    def test_computeTransformationProjective(self):
        points = np.array([[1,1,1], [2,1,1], [1.5,2,1], [1.5,3,1], [3,2, 1]])
        
        projectiveTransformation = np.identity(3)
        projectiveTransformation[0, 0] = 3
        projectiveTransformation[1, 1] = 5
        projectiveTransformation[:, 2] = [2, 5, 1]
        projectiveTransformation[2, :] = [6, 11, 1]
        expectedPoints = [np.dot(projectiveTransformation, np.transpose(point)) for point in points]

        H = DLT.computeDLTTransformation(points, np.asarray(expectedPoints))

        measuredPoints = [np.dot(H, np.transpose(point)) for point in points]
        self.assertTrue(np.allclose(expectedPoints, measuredPoints))

if __name__ == "__main__":
    unittest.main()

