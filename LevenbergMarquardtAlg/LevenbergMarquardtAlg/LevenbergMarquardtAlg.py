import DLT

import numpy as np

def computeJacobian(func, A, x, dx):
    result = func(A, x)
    J = np.zeros((len(result), len(x)))
    for i in range(len(x)):
        x_dx = np.copy(x)
        x_dx[i] += dx
        result_dx = func(A, x_dx)
        J[:, i] = (result_dx - result) / dx

    return (result, J)

def computeErrorVector(A, h):
    e = np.matmul(A, np.transpose(h))
    return e.flatten()

def frobeniusNorm2(points):
    norm2 = np.sqrt(np.sum(np.square(points), axis = 0))
    return norm2

def iterativeMinimization_LMA(fromPoints, toPoints, maxIterations, errorThreshold):
    error, DLTTransformation = DLT.computeDLTTransformation(fromPoints, toPoints)
    A = DLT.matrixFromCorrespondences(fromPoints, toPoints)
    h = DLTTransformation.flatten()

    errVec, jacobian = computeJacobian(computeErrorVector, A, h, 1e-10)

    # initial value recommended: 1e-3
    lamda = 1e-3
    lamdaFactor = 2

    normErr = frobeniusNorm2(errVec)

    numofIterations = maxIterations
    while numofIterations > 0:
        if normErr < errorThreshold:
            break

        jTj = np.matmul(np.transpose(jacobian), jacobian)

        lamdaI = np.identity(jTj.shape[0]) * lamda
        jTerr = -np.matmul(np.transpose(jacobian), errVec)
        delta = np.matmul(np.linalg.pinv(jTj + lamdaI), jTerr)

        hNew = h + delta
        errVecNew = computeErrorVector(A, hNew)
        normErrNew = frobeniusNorm2(errVecNew)

        if normErrNew <= normErr:
            lamda /= lamdaFactor
            normErr = normErrNew
            h = hNew
            errVec = errVecNew
        else:
            lamda *= lamdaFactor

        numofIterations -= 1

    H = np.reshape(h, (3,3))
    H /= H[2,2]

    return (normErr, DLTTransformation, H)

if __name__ == "__main__":
    pass
