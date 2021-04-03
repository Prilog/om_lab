import math


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


def gradient(func, grad, step_func, eps):
    result = Statistics()
    old = 0
    l = step_func(lambda x: func(old - x * grad(old)), -1e6, 1e6, 1e-6).result
    new = old - l * grad(old)
    result.traectory = [old, new]
    result.iterations += 1
    result.computations += 2
    while abs(old-new) > eps:
        l = step_func(lambda x: func(new - x * grad(new)), -1e6, 1e6, 1e-6).result
        old, new = new, new - l * grad(new)
        result.traectory.append(new)
        result.iterations += 1
        result.computations += 1
    result.result = new
    return result


print(gradient(func=lambda x: (x + 1) ** 2,
               grad=lambda x: 2 * (x + 1),
               step_func=dichotomy,
               eps=1e-6).traectory)
