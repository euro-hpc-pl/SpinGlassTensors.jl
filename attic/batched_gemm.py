from timeit import timeit
import numpy as np 
from time import time

x = 32
y = 256 

a = np.random.rand(y, x, x)
b = np.random.rand(y, x, x)

if __name__ == "__main__":

    t1 = time()
    d = a @ b
    t2 = time()
    print(t2-t1)


    t1 = time()
    c = np.zeros_like(d)
    for i in range(0, c.shape[0]):
        c[i, :, :] = a[i, :, :] @ b[i, :, :]
    t2 = time()
    print(t2-t1)

    assert np.array_equal(c, d)