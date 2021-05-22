import math
import numpy as np


EPS = 0.0000001
INF = 1000000000.0


class LinearProblem:
    def __init__(self, func, limitations, initial):
        self.func = func
        self.limitations = limitations
        self.initial = initial
        self.n = func.shape[0]

    def evaluate(self, x):
        return x.dot(self.func)


def simplexMethod(problem):
    # build simplex matrix
    functional = np.concatenate((problem.func, np.array([0])))
    simplexMatrix = np.vstack((problem.limitations, functional))

    print(simplexMatrix)

    # fit simplex matrix to initial value
    used = {}
    for v in range(problem.n):
        if problem.initial[v] == 1:
            currentRow = 0
            for row in range(simplexMatrix.shape[0]):
                if row not in used and simplexMatrix[row][v] != 0:
                    currentRow = row
                    break
            simplexMatrix[currentRow] = simplexMatrix[currentRow] / simplexMatrix[currentRow][v]
            for row in range(simplexMatrix.shape[0]):
                if row == currentRow:
                    continue
                simplexMatrix[row] = simplexMatrix[row] - (simplexMatrix[currentRow] * simplexMatrix[row][v])
            used[currentRow] = True

    print(simplexMatrix)

    # initiate positions
    positions = []
    for i in range(problem.initial.shape[0]):
        if problem.initial[i] == 1:
            positions.append(i)

    def checkOptimal():
        # all values in a last row must be positive
        for i in range(problem.n):
            if simplexMatrix[-1][i] > EPS:
                return False
        return True

    def getLeadingRow(lc):
        value = INF
        result = 0
        leadingColumnValue = simplexMatrix[:, lc]
        lastColumnValue = simplexMatrix[:, -1]
        for i in range(simplexMatrix.shape[0]):
            if leadingColumnValue[i] > EPS and lastColumnValue[i] / leadingColumnValue[i] < value:
                value = lastColumnValue[i] / leadingColumnValue[i]
                result = i
        return result

    while not checkOptimal():
        print(simplexMatrix)
        leadingColumn = np.argmax(simplexMatrix[-1][:-1])
        leadingRow = getLeadingRow(leadingColumn)
        leadingElement = simplexMatrix[leadingRow][leadingColumn]
        print(leadingRow, leadingColumn)

        simplexMatrix[leadingRow] = simplexMatrix[leadingRow] / leadingElement
        for i in range(simplexMatrix.shape[0]):
            if i != leadingRow:
                simplexMatrix[i] = simplexMatrix[i] - simplexMatrix[leadingRow] * simplexMatrix[i][leadingColumn]
        break

test1 = LinearProblem(
    np.array([-6, -1, -4, 5], dtype=float),
    np.array([
        [3, 1, -1, 1, 4],
        [5, 1, 1, -1, 4]
    ], dtype=float),
    np.array([1, 0, 0, 1])
)

simplexMethod(problem=test1)
