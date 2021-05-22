import math
import numpy as np


EPS = 0.0000001
INF = 1000000000.0


class LinearProblem:
    def __init__(self, func, limitations, initial):
        self.func = func.astype(float)
        self.limitations = limitations.astype(float)
        self.initial = initial
        self.n = func.shape[0]

    def evaluate(self, x):
        return x.dot(self.func)


def simplexMethod(problem):
    # build simplex matrix
    functional = np.concatenate((problem.func, np.array([0])))
    functional = -functional
    simplexMatrix = np.vstack((problem.limitations, functional))


    # fit simplex matrix to initial value
    used = {}
    for v in range(problem.n):
        if problem.initial[v] != 0:
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

    # initiate positions
    positions = []
    for i in range(problem.initial.shape[0]):
        if problem.initial[i] != 0:
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
        for i in range(simplexMatrix.shape[0] - 1):
            if leadingColumnValue[i] > EPS and lastColumnValue[i] / leadingColumnValue[i] < value:
                value = lastColumnValue[i] / leadingColumnValue[i]
                result = i
        return result

    while not checkOptimal():
        # order positions
        newPositions = []
        for i in range(simplexMatrix.shape[0] - 1):
            for j in positions:
                if simplexMatrix[i][j] > EPS:
                    newPositions.append(j)
        positions = newPositions

        # select leading stuff
        leadingColumn = np.argmax(simplexMatrix[-1][:-1])
        leadingRow = getLeadingRow(leadingColumn)
        leadingElement = simplexMatrix[leadingRow][leadingColumn]

        simplexMatrix[leadingRow] = simplexMatrix[leadingRow] / leadingElement
        for i in range(simplexMatrix.shape[0]):
            if i != leadingRow:
                simplexMatrix[i] = simplexMatrix[i] - simplexMatrix[leadingRow] * simplexMatrix[i][leadingColumn]
        positions[leadingRow] = leadingColumn

    result = np.zeros(problem.n)
    for i in range(len(positions)):
        result[positions[i]] = simplexMatrix[i][-1]
    return result, simplexMatrix[-1][-1]


test1 = LinearProblem(
    np.array([-6, -1, -4, 5]),
    np.array([
        [3, 1, -1, 1, 4],
        [5, 1, 1, -1, 4]
    ]),
    np.array([1, 0, 0, 1])
)

test2 = LinearProblem(
    np.array([-1, -2, -3, 1]),
    np.array([
        [1, -3, -1, -2, -4],
        [1, -1, 1, 0, 0]
    ]),
    np.array([0, 1, 1, 0])
)

test3 = LinearProblem(
    np.array([-1, -2, -1, 3, -1]),
    np.array([
        [1, 2, 0, 2, 1, 5],
        [1, 1, 1, 3, 2, 9],
        [0, 1, 1, 2, 1, 6]
    ]),
    np.array([0, 0, 1, 2, 1])
)

test4 = LinearProblem(
    np.array([-1, -1, -1, 1, -1]),
    np.array([
        [1, 1, 2, 0, 0, 4],
        [0, -2, -2, 1, -1, -6],
        [1, -1, 6, 1, 1, 12]
    ]),
    np.array([1, 1, 2, 0, 0])
)

test5 = LinearProblem(
    np.array([-1, 4, -3, 10]),
    np.array([
        [1, 1, -1, -10, 4],
        [1, 14, 10, -10, 11]
    ]),
    np.array([1, 0, 1, 0])
)

test6 = LinearProblem(
    np.array([-1, 5, 1, -1, 0, 0]),
    np.array([
        [1, 3, 3, 1, 1, 0, 3],
        [2, 0, 3, -1, 0, 1, 4]
    ]),
    np.array([0, 0, 0, 0, 1, 1])
)

test7 = LinearProblem(
    np.array([-1, -1, 1, -1, 2]),
    np.array([
        [3, 1, 1, 1, -2, 10],
        [6, 1, 2, 3, -4, 20],
        [10, 1, 3, 6, -7, 30]
    ]),
    np.array([1, 0, 1, 1, 0])
)

tests = [test1, test2, test3, test4, test5, test6, test7]

for i in range(len(tests)):
    print("Test:", i + 1)
    vector, value = simplexMethod(tests[i])
    print("Variables:", vector)
    print("Value:", value)
    print("============================")
