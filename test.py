'''
Created on Jul 26, 2013

@author: Amin
'''

import numpy
import os
import csv
import math

D = 3

class filewriter(object):
    
    f = ""
    
    def __init__(self, path):
        self.f = open(path,"w")
    
    def writeText(self, text):
        self.f.write(text)
    
    def close(self):
        self.f.close()
        
def main():
    x1 = {}
    x1["A"] = 2
    if "A" in x1.keys():
        print("T")
    print(x1) 
    return
    m = numpy.matrix('1 2 3; 3 4 5')
    s = numpy.matrix('2 3 3; 1 2 2')
    print(m / 2)
    
    return
    tr,ts = get_xyz_data("data\\Binned", "x")
    E, P, Pi, cent, gestureRecThreshold = load_model("data", "x")
    
    lik = pr_hmm(tr[0], P, E.transpose(), Pi)
    print(lik)
    return
    
def mat_div(m, x):
    z = numpy.zeros((len(m), len(m[0])))
    for i in range(len(m)):
        for j in range(len(m[0])):
            z[i,j] = m[i,j] / x
    return z 

def get_xyz_data (path, name):
    fx = open(path + os.sep + name + '.csv', 'r')
    cx = csv.reader (fx, delimiter=',', quotechar='|')
    x = numpy.asarray(list(cx), dtype = 'float')
    
    fx.close()
    
    i, j = x.shape
    tr = x[:i/2,:]
    ts= x[i/2:,:]
    return (tr, ts)

def pr_hmm(o, a, b, pi):
    n = a.shape[1]
    T = len(o)
    print(o.shape)
#     print(o.shape,a.shape,b.shape,pi.shape,n,T)
    m = numpy.zeros(shape=(T, n))
#     print(m.shape)
    # it uses forward algorithm to compute the probability
    for i in range(n):  # initilization
        m[0, i] = b[i, o[0]] * pi[i]
        
    for t in range(T - 1):  # recursion
        for j in range(n):
            z = 0
            for i in range(n):
                z = z + a[i, j] * m[t, i]
            m[t + 1, j] = z * b[j, o[t + 1]]
        
    p = 0
    
    for i in range(n):  # termination
        p = p + m[T - 1, i]
    
    p = math.log(p)
    return p

def load_model(path, name):
    Ef = open(path + os.sep + "model" + os.sep + name + ".csv", "r")
    ar = csv.reader (Ef, delimiter=',', quotechar='|')
    count = 0
    for row in ar:
        if count == 0:
            i = 0
            for item in row:
                if i == 0:
                    N = int(item) # Number of states
                else:
                    M = int(item) # Number of symbols
                    E = numpy.empty(shape = (N, M))
                    P = numpy.empty(shape = (M, M))
                    Pi = numpy.empty(M)
                    cent = numpy.empty(shape = (N, 3))
                i = i + 1
        elif count <= N:
            i = 0
            for item in row:
                E[count - 1, i] = float(item)
                i = i + 1
        elif count > N and count <= N + M:
            i = 0
            for item in row:
                P[count - N - 1, i] = float(item)
                i = i + 1
        elif count == N + M + 1:
            Pi = numpy.asarray(list(row), dtype = 'float')
        elif count > N + M + 1 and count < N + M + N + 2:
            i = 0
            for item in row:
                cent[count - N - M - 2, i] = float(item)
                i = i + 1
        elif count == N + M + N + 2:
            for item in row:
                thr = float (item)
        count = count + 1
    Ef.close()

    return (E, P, Pi, cent, thr)
    

if __name__ == '__main__':
    main()
    