import timeit
import os
import sys
import csv
import numpy
import math
from turtledemo import clock

p = []
def main ():
    
    
    for i in range(1,-2,-1):
        for j in range(1,-2,-1):
            for k in range(1,-2,-1):
                p.append(normalize_vector(numpy.array([i, j, k])))
    print(p)
    return
    x = numpy.array([1, 2, 3])
    y = numpy.array([-2, 3, -4])

    center = numpy.array([0, 0, 0])
    
    p.append (numpy.array([1, 1, 1]))
    p.append (numpy.array([1, 1, 0]))
    p.append (numpy.array([1, 1, -1]))
    p.append (numpy.array([1, 0, 1]))
    p.append (numpy.array([1, 0, 0]))
    p.append (numpy.array([1, 0, -1]))
    p.append (numpy.array([1, -1, 1]))
    p.append (numpy.array([1, -1, 0]))
    p.append (numpy.array([1, -1, -1]))

    p.append (numpy.array([0, 1, 1]))
    p.append (numpy.array([0, 1, 0]))
    p.append (numpy.array([0, 1, -1]))
    p.append (numpy.array([0, 0, 1]))
    p.append (numpy.array([0, 0, 0]))
    p.append (numpy.array([0, 0, -1]))
    p.append (numpy.array([0, -1, 1]))
    p.append (numpy.array([0, -1, 0]))
    p.append (numpy.array([0, -1, -1]))

    p.append (numpy.array([-1, 1, 1]))
    p.append (numpy.array([-1, 1, 0]))
    p.append (numpy.array([-1, 1, -1]))
    p.append (numpy.array([-1, 0, 1]))
    p.append (numpy.array([-1, 0, 0]))
    p.append (numpy.array([-1, 0, -1]))
    p.append (numpy.array([-1, -1, 1]))
    p.append (numpy.array([-1, -1, 0]))
    p.append (numpy.array([-1, -1, -1]))
    
#     for i in range(len(p)):
#         tmp_p = p[i]
#         p[i] = normalize_vector(p[i])
#         print(tmp_p, p[i])
#   
    '''
    v = numpy.array([ -0.76, -0.11,  0.64])  
    v = numpy.array([ 0.35517656, -0.01604688,  0.93466149])
    v = numpy.array([ 0.14359442, -0.85306931, -0.50165066])
    
    v = normalize_vector(v)
    print("Recursion:", get_direction(v))
    print("Maximization:", get_direction_dot(v, p))
    
    return
    
    for pn in p:
        rec_dir = get_direction(pn)
        dir = get_direction_dot(pn, p)
        print(dir, rec_dir)
    ''' 
    print(timeit.timeit(stmt = "test_direction()", setup = "from __main__ import test_direction", number = 10000))
    print(timeit.timeit(stmt = "test_direction_dot()", setup = "from __main__ import test_direction_dot", number = 10000))

def test_direction():
    vec = numpy.random.randn(3)
    vec = normalize_vector(vec)
    return get_direction(vec)

def test_direction_dot():
    vec = numpy.random.randn(3)
    vec = normalize_vector(vec)
    return get_direction_dot(vec, p)
    
def test_timer():
    test_loop = 1000
    test_fail = 0
    print("Testing a sequence of %d with random vectors..." % (test_loop))
    for i in range(test_loop):
        vec = numpy.random.randn(3)
        vec = normalize_vector(vec)
        rec_dir = get_direction(vec)
        dir = get_direction_dot(vec, p)
        if rec_dir != dir:
            test_fail = test_fail + 1
            print("One fail for vector", vec, ". Recursion is %d and Dot Maximize is %d" %(rec_dir, dir))
    
    print("Testing finished with %d fail(s). The success rate is %.2f" %(test_fail, 100 * (1 - test_fail / test_loop)))

def normalize_vector(vec):
    nrm = numpy.linalg.norm(vec)
    if nrm == 0:
        return vec
    else:
        return vec / nrm
        
def get_direction(vec):
    if vec[0] < 0:
        if vec[1] < 0:
            return get_direction_dot(vec, p[24:]) + 24
        else:
            return get_direction_dot(vec, p[18:24]) + 18
    else:
        return get_direction_dot(vec, p[:24])
        
            
    
def get_direction_dot(vec, p):
    vec = normalize_vector(vec)
    if numpy.linalg.norm(vec) == 0:
        return 13
    
    max_dist = -2
    max_dist_index = -1
    for i in range(len(p)):
        d = numpy.dot(vec, p[i])
        if d > max_dist:
            max_dist = d
            max_dist_index = i
    return max_dist_index
    
     
    
if __name__ == '__main__':
    main()