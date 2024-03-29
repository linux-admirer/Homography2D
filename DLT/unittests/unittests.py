import unittest

import numpy as np

import DLT

def normalizePoints(points):
    points[:, 0] /= points[:, 2]
    points[:, 1] /= points[:, 2]
    points[:, 2] /= points[:, 2]
    return points

class TestDLT(unittest.TestCase):
    def test_computeCentroid(self):
        points = np.array([[1,2], [3,4]])
        centroid = DLT.computeCentroid(points)
        self.assertTrue(np.array_equal(centroid, [2., 3.]))

    def test_computeTransformation(self):
        points = np.array([[1, 1, 1], [1, -1, 1], [-1, -1 , 1], [-1, 1, 1]])
        e, H = DLT.computeDLTTransformation(points, points)
        
        expectedTransformation = np.identity(3)
        self.assertTrue(np.allclose(H, expectedTransformation))

    def test_computeTransformationForSamePoints(self):
        points = np.array([[1,1,1], [2,1,1], [1.5,2,1], [1.5,3,1], [3,2, 1]])
        e, H = DLT.computeDLTTransformation(points, points)
        expectedTransformation = np.identity(3)
        self.assertTrue(np.allclose(H, expectedTransformation))

    def test_computeTransformationAffine(self):
        points = np.array([[1,1,1], [2,1,1], [1.5,2,1], [1.5,3,1], [3,2, 1]])
        
        affineTransformation = np.identity(3)
        affineTransformation[0, 0] = 3
        affineTransformation[1, 1] = 5
        affineTransformation[:, 2] = [2, 5, 1]
        expectedPoints = [np.dot(affineTransformation, np.transpose(point)) for point in points]

        e, H = DLT.computeDLTTransformation(points, np.asarray(expectedPoints))

        measuredPoints = [np.dot(H, np.transpose(point)) for point in points]
        self.assertTrue(np.allclose(expectedPoints, measuredPoints))

    def test_computeTransformationProjective(self):
        points = np.array([[1,1,1], [2,1,1], [1.5,2,1], [1.5,3,1], [3,2, 1]])
        
        projectiveTransformation = np.identity(3)
        projectiveTransformation[0, 0] = 3
        projectiveTransformation[1, 1] = 5
        projectiveTransformation[0, 1] = 0.01
        projectiveTransformation[1, 0] = 0.5
        projectiveTransformation[:2, 2] = [2, 5]
        projectiveTransformation[2, :2] = [6, 11]
        expectedPoints = np.asarray(
            [np.dot(projectiveTransformation, np.transpose(point)) for point in points])

        expectedPoints = normalizePoints(expectedPoints)

        e, H = DLT.computeDLTTransformation(points, np.asarray(expectedPoints))
        print("Algebraic error: %s" %e)

        measuredPoints = np.asarray([np.dot(H, np.transpose(point)) for point in points])
        measuredPoints = normalizePoints(measuredPoints)

        self.assertTrue(np.allclose(expectedPoints, measuredPoints, atol=0.1))

    def test_computeTransformationProjectiveWithGaussianNoise(self):
        points = np.array([[1,1,1], [2,1,1], [1.5,2,1], [1.5,3,1], [3,2, 1]])
        
        projectiveTransformation = np.identity(3)
        projectiveTransformation[0, 0] = 3
        projectiveTransformation[1, 1] = 5
        projectiveTransformation[0, 1] = 0.01
        projectiveTransformation[1, 0] = 0.5
        projectiveTransformation[:2, 2] = [2, 5]
        projectiveTransformation[2, :2] = [6, 11]
        expectedPoints = np.asarray(
            [np.dot(projectiveTransformation, np.transpose(point)) + np.random.normal(0, 0.1) for point in points])

        expectedPoints = normalizePoints(expectedPoints)

        e, H = DLT.computeDLTTransformation(points, np.asarray(expectedPoints))
        print("Algebraic error: %s" %e)

        measuredPoints = np.asarray([np.dot(H, np.transpose(point)) for point in points])
        measuredPoints = normalizePoints(measuredPoints)

        self.assertTrue(np.allclose(expectedPoints, measuredPoints, atol=0.1))

    def test_computeTransformationCollinearPoints(self):
        points = np.array([[1,1,1], [2,2,1], [3,3,1], [4,4,1]])
        
        projectiveTransformation = np.identity(3)
        expectedPoints = [np.dot(projectiveTransformation, np.transpose(point)) for point in points]

        with self.assertRaises(Exception):
            DLT.computeDLTTransformation(points, np.asarray(expectedPoints))

if __name__ == "__main__":
    unittest.main()
