import os
import numpy as np

def computeCentroid(points):
    pointSum = np.sum(points, axis = 0)
    dim = points.shape[0]
    centroid = pointSum / dim
    return centroid

def averageDistance2D(points):
    distancePoints = [np.sqrt(np.square(point[0]) + np.square(point[1])) for point in points]
    distance = np.sum(distancePoints)
    return (distance/points.shape[0])

def normalizePoints2D(points):
    centroid = computeCentroid(points)

    # Average distance of points from origin must be sqrt(2)
    avgDistance = averageDistance2D(points)
    scale = np.sqrt(2) / avgDistance

    transformation = np.identity(3)
    transformation = np.multiply(transformation, scale)
    transformation[:, 2] = -centroid
    transformation[2,2] = 1

    transformedPoints = [np.dot(transformation, np.transpose(point)) for point in points]
    return (transformation, np.asarray(transformedPoints))

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

    # TODO: computation of U can be optimized using bidiagonilization SVD
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

    estimatedPoints = np.asarray([np.matmul(transformation, point) for point in fromPoints])
    avgEstDist = averageDistance2D(estimatedPoints)
    
    # fromPoints are normalized such that the average distance from origin is sqrt(2).
    # The estimated points must be scaled such that average distance is also sqrt(2)
    return np.sqrt(2) / avgEstDist


def computeDLTTransformation(fromPoints, toPoints):
    fromTransformation, fromPointsNormalized = normalizePoints2D(fromPoints)
    toTransformation, toPointsNormalized = normalizePoints2D(toPoints)

    error, H = compute2DDLTMatrixHomogeneous(fromPointsNormalized, toPointsNormalized)
    print("Algebraic error = %s" %error)

    transformationOrig = np.matmul(np.linalg.inv(toTransformation), H)
    transformationOrig = np.matmul(transformationOrig, fromTransformation)
    transformationOrig /= transformationOrig[2,2]

    return transformationOrig

if __name__ == "__main__":
    points = np.array([[1, 1, 1], [1, -1, 1], [-1, -1 , 1], [-1, 1, 1]])
    computeDLTTransformation(points, points)
