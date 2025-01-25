import numpy as np


def calucalate_distance(x1, y1, x2, y2, is_prime, it_tenth):
    if is_prime:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    else:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * 1.1
