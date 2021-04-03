import math

# todo: add fixed number of iterations to all functions


class Statistics:
    def __init__(self):
        self.iterations = 0
        self.computations = 0
        self.answer = None
        self.result = None


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
        if less(f1, f2, eps):
            r = mid
        else:
            l = mid
        result.computations += 2
        result.iterations += 1
    result.result = l
    return result


# todo: can be optimised, by reducing number of computations to 1
def golden(f, l, r, eps):
    fi = (1 + math.sqrt(5)) / 2
    result = Statistics()
    x1 = r - (r - l) / (fi + 1)
    x2 = l + (r - l) / (fi + 1)
    f1 = f(x1)
    f2 = f(x2)
    while r - l > eps:
        print(l, r, f1, f2)
        if less(f1, f2, eps):
            r = x2
            x2 = x1
            f2 = f1
            x1 = l + (r - l) / (fi + 1)
            f1 = f(x1)
        else:
            l = x1
            x1 = x2
            f1 = f2
            x2 = r - (r - l) / (fi + 1)
            f2 = f(x2)
        result.computations += 1
        result.iterations += 1
    print(l, r)
    result.result = (l + r) / 2
    return result


# todo: can be optimised like golden
def fibonacci(f, l, r, eps):
    fi = (1 + math.sqrt(5)) / 2
    result = Statistics()
    while r - l > eps:
        # todo: fix x1, x2, accoarding to fibonacci numbers. See: shorturl.at/djAHY
        x1 = r - (r - l) / fi
        x2 = l + (r - l) / fi
        f1 = f(x1)
        f2 = f(x2)
        if f1 - f2 > -eps:
            l = x1
        else:
            r = x2
        result.computations += 2
        result.iterations += 1
    result.result = (l + r) / 2
    return result


print(golden(lambda x: (x + 1) ** 2, -100, 100, 0.000000001).result)
