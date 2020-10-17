from timeit import default_timer as timer
from functools import wraps
import statistics
import random

def timing(f):
    @wraps(f)
    def wrapper(*args, **kw):
        start = timer()
        result = f(*args, **kw)
        end = timer()
        return result, end-start
    return wrapper

def timing2(f):
    @wraps(f)
    def wrapper(*args, **kw):
        start = timer()
        iter_ = 0
        while True:
            result = f(*args, **kw)
            iter_ += 1
            t = timer()
            if t-start < 1000:
                break
        return result, (t-start)/iter_
    return wrapper

@timing2
def better_random(n):
    # Create vector with values
    vec = list(range(n))
    # Select random value, move it to the end of the vector
    for i in range(n,0,-1):
        vec.append(vec.pop(random.choice(vec[:i])))
    return vec


if __name__ == '__main__':
    iters = [better_random(1000) for _ in range(500)]
    # Display one permutation
    # print(iters[0][0])
    # Display average times
    print(statistics.mean([i[1] for i in iters]))

# Avg timings | permutation size = 1000, iterations 500
# basic - 0.00267
# with numba - 
#
#