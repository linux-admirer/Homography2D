import unittest

import numpy as np

import LevenbergMarquardtAlg as LMA
import DLT

def equation(A, x):
    v = np.matmul(A, np.transpose(x))
    return v.flatten()

class TestLMA(unittest.TestCase):
    def test_computeJacobian(self):
        # linear equation 1 = x0 + 2 * x1
        # linear equation 2 = 2 * x0 + 3 * x1
        linearEq = np.asarray([[1, 2], [2, 3]])
        jacobian = LMA.computeJacobian(equation, linearEq, np.asarray([1,2], dtype='float'), 1e-10)
        print(jacobian)

    def test_computeTransformationAffine(self):
        points = np.array([[1,1,1], [2,1,1], [1.5,2,1], [1.5,3,1], [3,2, 1]])

        affineTransformation = np.identity(3)
        affineTransformation[0, 0] = 3
        affineTransformation[1, 1] = 3
        affineTransformation[:, 2] = [2, 5, 1]
        expectedPoints = np.asarray([np.dot(affineTransformation, np.transpose(point)) for point in points])

        err, DLTTransformation, H = LMA.iterativeMinimization_LMA(points, expectedPoints, 100, 1e-3)

        measuredPoints = np.asarray([np.dot(H, np.transpose(point)) for point in points])
        DLT._homogenizePoints(measuredPoints)
        self.assertTrue(np.allclose(expectedPoints, measuredPoints, atol=0.1))

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

        expectedPoints = DLT._homogenizePoints(expectedPoints)

        err, DLTTransformation, H = LMA.iterativeMinimization_LMA(points, np.asarray(expectedPoints), 1000, 1e-3)

        measuredPoints = np.asarray([np.dot(H, np.transpose(point)) for point in points])
        DLT._homogenizePoints(measuredPoints)

        self.assertTrue(np.allclose(expectedPoints, measuredPoints, atol=0.1))

if __name__ == "__main__":
    unittest.main()
