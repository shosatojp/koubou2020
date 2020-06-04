import numpy as np


def AND(x):
    x = np.array(x)
    w = np.array([0.5, 0.5])
    b = -0.7
    y = np.sum(w*x) + b
    return y >= 0

# (0, 0) -> False
# (1, 0) -> False
# (0, 1) -> False
# (1, 1) -> True

def NAND(x):
    x = np.array(x)
    w = np.array([-1, -1])
    b = 1.4
    y = np.sum(w*x) + b
    return y >= 0

# (0, 0) -> True
# (1, 0) -> True
# (0, 1) -> True
# (1, 1) -> False

def OR(x):
    x = np.array(x)
    w = np.array([1, 1])
    b = -0.7
    y = np.sum(w*x) + b
    return y >= 0

# (0, 0) -> False
# (1, 0) -> True
# (0, 1) -> True
# (1, 1) -> True

def XOR(x):
    x = np.array(x)
    return AND([NAND(x), OR(x)])

# (0, 0) -> False
# (1, 0) -> True
# (0, 1) -> True
# (1, 1) -> False

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        out = NAND(xs)
        print(str(xs) + " -> " + str(out))
