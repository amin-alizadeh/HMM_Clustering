from dataflow import dataflow
import os
import numpy
from RecordingAnalyzer import get_xyz_data
import math

def main():
    print(math.log(2.14727368714e-64))
    return
    path = "data" + os.sep + "unitytrain"
    name = "circle-r-cw"
    dtf = dataflow(path, name)
    
    (E, P, Pi, cent, thr) = dtf.load_model()
    testFile = "2014-3-31--16-50-44.csv"
    alltests = get_xyz_data(path + os.sep + "circle-r-cw" + os.sep + testFile)
    joint = "Hand_R"
    
    test = alltests[joint]
    clusters = numpy.empty(len(test))
    #f = open ("output.txt", "wt")
    for i in range(len(test)):
        clusters[i] = get_point_cluster(test[i,:], cent)
       # f.write(str(clusters[i]) + "\n")
        
    
    score = pr_hmm(clusters, P, E, Pi)
    print(score)
  #  f.write(str(score))
   # f.close()
    
def pr_hmm(clusters, P, E, pi):
    # Clusters, P, E , Pi
    # clusters       , P, E , pi
    f = open("pr_hmm.txt", "wt")
    
    n = P.shape[1] # Which is M, the number of hidden symbols
    T = len(clusters) # The number of clustered observed data
    #f.write("n= " + str(n) + " T = " + str(T) + "\n")
    m = numpy.zeros(shape=(T, n))
    # it uses forward algorithm to compute the probability
    for i in range(n):  # initilization
        m[0, i] = E[int(clusters[0]), i] * pi[i]
        #f.write("m[0, i] = " + str(m[0, i]) + ", ")
   # f.write("\n")
        
    for t in range(T - 1):  # recursion
        for j in range(n):
            z = 0
           # f.write("i loop to n begins\n")
            for i in range(n):
                z = z + P[i, j] * m[t, i]
               # f.write("z= " + str(z) + "\n")
           # f.write("i loop to n ends\n")
            m[t + 1, j] = z * E[int(clusters[t + 1]), j]
            #f.write("m[t+1, j] = " + str(m[t + 1, j]) + ",")
       # f.write("\n")
    
    for row in m:
        for el in row:
            f.write(str(el) + ", ")
        f.write("\n")
    f.close()
    Pr = 0
    
    for i in range(n):  # termination
        Pr = Pr + m[T - 1, i]
    
    if Pr != 0:
        Pr = math.log(Pr)
    else:
        Pr = float('-inf')
    
    #f.close() 
    return Pr    
    
    
def get_point_cluster (point, centroids):
    tmp = numpy.empty (len(centroids))
    for i in range(len(centroids)):
        tmp[i] = math.sqrt((centroids[i, 0] - point[0]) ** 2 + \
                           (centroids[i, 1] - point[1]) ** 2 + \
                           (centroids[i, 2] - point[2]) ** 2)
    
    idx, I = min((val, idx) for (idx, val) in enumerate(tmp))
    
    return I
    
    
if __name__ == '__main__':
    main()