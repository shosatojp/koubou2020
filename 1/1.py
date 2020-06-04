import numpy as np


def AND(x):
    x = np.array(x)
    w = np.array([0.5, 0.5])
    b = -0.7
    y = np.sum(w*x) + b
    return y >= 0


def NAND(x):
    x = np.array(x)
    w = np.array([-1, -1])
    b = 1.4
    y = np.sum(w*x) + b
    return y >= 0


def OR(x):
    x = np.array(x)
    w = np.array([1, 1])
    b = -0.7
    y = np.sum(w*x) + b
    return y >= 0


def XOR(x):
    x = np.array(x)
    return AND([NAND(x), OR(x)])


if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        out = XOR(xs)
        print(str(xs) + " -> " + str(out))
