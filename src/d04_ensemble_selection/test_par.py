from joblib import parallel_backend, Parallel, delayed
from math import sqrt, exp


def fibonacci(n):
    if n==0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# with parallel_backend("threading", n_jobs=8):
results = Parallel(n_jobs=8)(delayed(fibonacci)(i) for i in range(int(1e2)))

print(results[-20:])
