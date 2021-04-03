import math
import numpy


class Statistics:
    def __init__(self):
        self.iterations = 0
        self.computations = 0
        self.answer = None
        self.result = None
        self.traectory = []


def greater(a, b, eps):
    return a - b > -eps


def less(a, b, eps):
    return a - b < eps


def dichotomy(f, l, r, eps):
    result = Statistics()
    while r - l > eps:
        mid = (l + r) / 2
        f1 = f(mid - eps)
        f2 = f(mid + eps)
        if f1 < f2:
            r = mid
        else:
            l = mid
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


def gradient(func, grad, step_func, eps):
    result = Statistics()

    def func_from_grads(dot):
        return lambda x: func(*[elem - x * grad[i](new) for i, elem in enumerate(dot)])

    def gradient_step(elem):
        l = step_func(func_from_grads(elem), -1e3, 1e3, 1e-6)
        return [elem[i] - l.result * grad[i](elem) for i in range(len(elem))]
        result.iterations += 1
        result.computations += l.computations

    old = [0, 0, 1]
    new = [0, 0, 0]
    new = gradient_step(old)
    result.traectory = [old[:], new[:]]

    while any(abs(cur - prev) > eps for cur, prev in zip(old, new)):
        l = step_func(func_from_grads(new), -1e3, 1e3, 1e-6)
        old = new[:]
        new = gradient_step(old)
        result.traectory.append(new[:])
    result.result = new
    return result


print(gradient(func=lambda x, y, z: (x - 5 + y) ** 2 + (z + 3) ** 2,
               grad=[lambda p: 2 * (p[0] - 5 + p[1]), lambda p: 2 * (p[0] - 5 + p[1]), lambda p: 2 * (p[2] + 3)],
               step_func=linear,
               eps=1e-6).traectory)
