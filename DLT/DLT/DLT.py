import os
import numpy as np

def computeCentroid(points):
    pointSum = np.sum(points, axis = 0)
    dim = points.shape[0]
    centroid = pointSum / dim
    return centroid

def averageDistance(points):
    points = np.square(points)
    sum = np.sum(points, axis=1)
    distancePoints = np.sqrt(sum)
    distance = np.sum(distancePoints)
    return (distance/points.shape[0])

def scaleFactor(points):
    avgDistance = averageDistance(points)
    return np.sqrt(2) / avgDistance

def normalizePoints(points):
    centroid = computeCentroid(points)
    # Translate points
    translatedPoints = np.subtract(points, centroid)

    # Average distance of points from origin must be sqrt(2)
    avgDistance = averageDistance(points)
    scale = np.sqrt(2) / avgDistance
    scaledPoints = np.multiply(points, scale)
    return scaledPoints

def compute2DDLTMatrixHomogeneous(fromPoints, toPoints):
    if not fromPoints.shape[1] == 3 or not toPoints.shape[1] == 3:
        raise Exception("Points must of 3 dimensions")

    if fromPoints.shape[0] != toPoints.shape[0]:
        raise Exception("Points must of the same size")

    if fromPoints.shape[0] < 4 or toPoints.shape[0] < 4:
        raise Exception("Need atleast 4 point correspondences")

    # X_expected cross H * X_measured = 0. This can be transformed to AH = 0.
    # A is a 2*number of points X 9 matrix
    A = np.zeros(shape=(2*fromPoints.shape[0], 9))
    rowIndex = 0
    for index, point in enumerate(toPoints):
        scaleX_from = point[0] * fromPoints[index]
        scaleY_from = point[1] * fromPoints[index]
        scaleW_from = point[2] * fromPoints[index]

        row1 = np.zeros(3)
        row1 = np.append(row1, -scaleW_from, axis = 0)
        row1 = np.append(row1, scaleY_from, axis = 0)
        A[rowIndex] = row1
        rowIndex = rowIndex + 1

        row2 = scaleW_from
        row2 = np.append(row2, np.zeros(3), axis = 0)
        row2 = np.append(row2, -scaleX_from, axis = 0)
        A[rowIndex] = row2
        rowIndex = rowIndex + 1

    # computation of U can be optimized using bidiagonilization SVD
    U, S, VT = np.linalg.svd(A, full_matrices=True)

    minIndex = np.where(S == np.amin(S))[0]
    # if only 4 point correspondences are given, there will only be 8 singular values.
    # The singular vector will be last row of VT
    if minIndex < 9:
        minIndex = 8
    h = VT[minIndex]

    algError = np.matmul(A, np.transpose(h))
    algErrpr = np.square(algError)
    algError = np.sum(algError)
    return (algError, np.reshape(h, (3,3)))

def estimateScaleNormalized(fromPoints, transformation):
    estimatedPoints = np.zeros(shape = fromPoints.shape)
    index = 0
    for point in fromPoints:
        estimatedPoints[index] = np.matmul(transformation, point)
        index = index + 1

    avgEstDist = averageDistance(estimatedPoints)
    
    # fromPoints are normalized such that the average distance from origin is sqrt(2).
    # The estimated points must be scaled such that average distance is also sqrt(2)
    return np.sqrt(2) / avgEstDist


def computeDLTTransformation(fromPoints, toPoints):
    fromPointsNormalized = normalizePoints(fromPoints)
    toPointsNormalized = normalizePoints(toPoints)

    error, H = compute2DDLTMatrixHomogeneous(fromPointsNormalized, toPointsNormalized)
    print("Algebraic error = %s" %error)

    # DLT transformation is done only upto scale. Scale must be calculated separately.
    scaleEstimated = estimateScaleNormalized(fromPointsNormalized, H)
    H = H * scaleEstimated
    return H

if __name__ == "__main__":
    points = np.array([[1, 1, 1], [1, -1, 1], [-1, -1 , 1], [-1, 1, 1]])
    computeDLTTransformation(points, points)

