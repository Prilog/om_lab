import math
import numpy as np
from datetime import datetime


class Statistics:
    def __init__(self):
        self.iterations = 0
        self.computations = 0
        self.answer = None
        self.result = None
        self.traectory = []
        self.deltas = []


def dichotomy(f, l, r, eps):
    result = Statistics()
    delta = r - l
    while r - l > eps:
        mid = (l + r) / 2
        f1 = f(mid - eps)
        f2 = f(mid + eps)
        if f1 < f2:
            r = mid
        else:
            l = mid
        result.deltas.append((r - l) - delta)
        delta = r - l
        result.computations += 2
        result.iterations += 1
    result.result = l
    return result


def golden(f, l, r, eps):
    fi = (1 + math.sqrt(5)) / 2
    result = Statistics()
    x1 = r - (r - l) / fi
    x2 = l + (r - l) / fi
    f1 = f(x1)
    f2 = f(x2)
    result.computations += 2
    delta = r - l
    while r - l > eps:
        if f1 < f2:
            r = x2
            x2 = x1
            f2 = f1
            x1 = r - (r - l) / fi
            f1 = f(x1)
        else:
            l = x1
            x1 = x2
            f1 = f2
            x2 = l + (r - l) / fi
            f2 = f(x2)
        result.deltas.append((r - l) - delta)
        delta = r - l
        result.computations += 1
        result.iterations += 1
    result.result = (l + r) / 2
    return result


def fibonacci(f, l, r, eps):
    n = 100
    fib = [0, 1, 1]
    for i in range(n - 2):
        fib.append(fib[-1] + fib[-2])
    result = Statistics()
    x1 = l + (r - l) * (fib[n - 2] / fib[n])
    x2 = l + (r - l) * (fib[n - 1] / fib[n])
    f1 = f(x1)
    f2 = f(x2)
    delta = r - l
    result.computations += 2
    while n > 1:
        n -= 1
        if f1 > f2:
            l = x1
            x1 = x2
            x2 = l + (r - l) * (fib[n - 1] / fib[n])
            f1 = f2
            f2 = f(x2)
        else:
            r = x2
            x2 = x1
            x1 = l + (r - l) * (fib[n - 2] / fib[n])
            f2 = f1
            f1 = f(x1)
        result.deltas.append((r - l) - delta)
        delta = r - l
        result.computations += 1
        result.iterations += 1
    result.result = (l + r) / 2
    return result


def linear(f, l, r, eps):
    eps = 0.1
    val = f(l)
    res = l
    i = l + eps
    while i <= r:
        if f(i) < val:
            val = f(i)
            res = i
        i += eps
    result = Statistics()
    result.result = res
    return result


class QuadraticFunction:
    """
    Format:
    F = x * A * xt + B * xt + c
    D = 1 * A * xt + x * A * 1t + b * 1t
    """

    def __init__(self, A, B, c):
        self.A = A
        self.B = B
        self.c = c
        self.n = A.shape[0]

    def func(self, x):
        return (x.dot(self.A).dot(x.transpose()) + self.B.dot(x.transpose()) + self.c)[0][0]

    def derivative(self, x):
        ones = np.array([np.ones(self.n)])
        return (ones.dot(self.A).dot(x.transpose()) + x.dot(self.A).dot(ones.transpose()) + self.B.dot(ones.transpose()))[0][0]

    def partial_derivative(self, x, idx):
        ones = np.array([np.zeros(self.n)])
        ones[0][idx] = 1
        x = np.array([x])
        return (ones.dot(self.A).dot(x.transpose()) + x.dot(self.A).dot(ones.transpose()) + self.B.dot(ones.transpose()))[0][0]

    def __str__(self):
        result = ""
        for i in range(self.n):
            for j in range(self.n):
                if self.A[i][j] == 0:
                    continue
                result += str(self.A[i][j])
                if i == j:
                    result += "x[" + str(i) + "]^2 + "
                else:
                    result += "x[" + str(i) + "]x[" + str(j) + "] + "
        for i in range(self.n):
            if self.B[i] == 0:
                continue
            result += str(self.B[i]) + "x[" + str(i) + "] + "
        result += str(self.c)
        return result


def build_random_QF(condition_number, size, seed=None):
    if seed is None:
        seed = condition_number * size + 10
    np.random.seed(seed)
    A = np.zeros(shape=(size, size), dtype=float)
    diagonal = np.random.uniform(low=1, high=condition_number, size=size)
    diagonal[0] = 1
    diagonal[-1] = condition_number
    np.fill_diagonal(A, diagonal)
    B = np.random.uniform(low=1, high=condition_number, size=size)
    c = np.random.uniform(low=1, high=condition_number, size=1)
    return QuadraticFunction(A, B, c[0])


def gradient(func, n, step_func, eps):
    result = Statistics()

    def func_from_grads(dot):
        return lambda x: func.func(np.array([[elem - x * func.partial_derivative(dot, i) for i, elem in enumerate(dot)]]))

    def gradient_step(elem):
        l = step_func(func_from_grads(elem), -1e3, 1e3, 1e-6)
        result.iterations += 1
        result.computations += l.computations
        return [elem[i] - l.result * func.partial_derivative(elem, i) for i in range(len(elem))]

    old = [0] * n
    new = gradient_step(old)
    result.traectory = [old[:], new[:]]

    while any(abs(cur - prev) > eps for cur, prev in zip(old, new)):
        old = new[:]
        new = gradient_step(old)
        result.traectory.append(new[:])
    result.result = new
    return result


# print(gradient(func=lambda x, y, z: (x - 5 + y) ** 2 + (z + 3) ** 2,
#                grad=[lambda p: 2 * (p[0] - 5 + p[1]), lambda p: 2 * (p[0] - 5 + p[1]), lambda p: 2 * (p[2] + 3)],
#                step_func=linear,
#                eps=1e-6).traectory)

# A = np.array([[2.0, 3.0], [4.0, 5.0]])
# B = np.array([[2.0, 2.0]])
# c = 0.0
#
# t = QuadraticFunction(A, B, c)
# x = np.array([[1.0, 1.0]])
# print(t.func(x))
# xx = np.array([1.0, 1.0])
# print(t.partial_derivative(xx, 1))
# print(build_random_QF(10, 5))
# stats = gradient(func=build_random_QF(100, 10), n=10, step_func=fibonacci, eps=1e-6)
# print(stats.result, stats.iterations, stats.computations, stats.answer, stats.traectory)


def test_on_random_cn(cn_s, n, step_function):
    results = []
    for cn in cn_s:
        func = build_random_QF(cn, n)
        results.append(gradient(func, n, step_function, 1e-6))
    return results


print(list(map(lambda e: e.iterations, test_on_random_cn([1, 10, 100], 5, fibonacci))))
