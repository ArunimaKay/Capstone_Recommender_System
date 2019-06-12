from __future__ import print_function

import numpy as np


def zero_pad(X, seq_len):
    return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X])


def get_vocabulary_size(X):
    return max([max(x) for x in X]) + 1  # plus the 0th word


def fit_in_vocabulary(X, voc_size):
    return [[w for w in x if (w < voc_size)] for x in X]


def batch_generator(X, y, batch_size):
    """Primitive batch generator 
    """
    size = X.shape[0]
    X_copy = np.array(X.copy())
    y_copy = np.array(y.copy())
    indices = np.arange(size)
    #print(f"Indices: ({len(indices)} elemements) sample: {indices[:10]}")
    np.random.shuffle(indices)
    #print(f"X_copy ({len(X_copy)} elements) sample: {X_copy[:10]}")
    X_copy = X_copy[indices.astype(int)]
    #print(f"y_copy ({len(y_copy)} elements) sample: {y_copy[:10]}")
    y_copy = y_copy[indices.astype(int)]
    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            continue


if __name__ == "__main__":
    # Test batch generator
    gen = batch_generator(np.array(['a', 'b', 'c', 'd']), np.array([1, 2, 3, 4]), 2)
    for _ in range(8):
        xx, yy = next(gen)
        print(xx, yy)
