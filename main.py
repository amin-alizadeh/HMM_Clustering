'''
Created on Jun 6, 2013

@author: Amin
'''

import csv
import os
import numpy
import math

D = 3 # the number of dimensions to use: X, Y, Z
M = 12 # output symbols
N = 8 # states
LR = 2 # degree of play in the left-to-right HMM transition matrix 

train_gesture = 'x'
test_gesture = 'x'


def main ():
    '''trainfilename = 'data' + os.sep + 'train' + os.sep + train_gesture + '_x.csv'
    f = open (trainfilename, "r")
    lines = f.readlines()
    f.close
    print(lines)
    '''
    path = 'data' + os.sep 
    training = get_xyz_data(path + 'train', train_gesture);
    #testing = get_xyz_data(path + 'test',test_gesture);
    
    #print(len(training[0][:][:]))
    #print(training)
    #print(len(training))
    #print(training[2][59][9])
    for n in range(len(training[0][:][:])):
        for i in range(len(training[0][0][:])):
            for j in range(D):
                #print(n,i,j, training[j][n][i])
                pass
            
    '''
    ****************************************************
    Initializing
    ****************************************************
    '''
    gestureRecThreshold = 0 # set below
    
    centroids, NX = get_point_centroids(training,N,D)
    
#     print(len(training[0][0][:]))
    ATrainBinned = get_point_clusters(training,centroids,D)
    print(ATrainBinned)
    
    #ATestBinned = get_point_clusters(testing,centroids,D)
    

def get_point_clusters(data,centroids,D):
    print(len(data[0][0][:]))
    XClustered = numpy.zeros(shape = (len(data[0][0][:]),1))
    K = len(centroids[:,0])
    

    for n in range(len(data[0][:][:])):
        for i in range(len(data[0][0][:])):
            temp = numpy.zeros(shape = (K, 1))
            for j in range(K):
                if D == 3:
                    temp[j] = math.sqrt((centroids[j,0] - data[0][n][i])**2+(centroids[j,1] - data[1][n][i])**2+(centroids[j,2] - data[2][n][i])**2)
            
            idx, I = min((val, idx) for (idx, val) in enumerate(temp))
            XClustered[n,1] = I # TO FINISH AND RETURN CORRECTLY
    print(temp, idx, I)
    return XClustered
    
def get_point_centroids(data,K,D):
    
    mean = numpy.zeros(shape = (len(data[0][:][:]),D)) # 60 x 3
    
    for n in range(len(data[0][:][:])): #60 number of rows
        for i in range(len(data[0][0][:])): #10 number of columns
            for j in range(D): #3
                mean[n][j] = mean[n][j] + data[j][n][i]
                #print(n,i,j)
                
    Nmeans = mat_div (mean, len(data[0][0][:]))
    centroids, points, idx = kmeans(Nmeans, K)
    
    K = len(centroids[0])
    #print(len(Nmeans[0]))
    return (centroids,K)

'''
usage
function[centroid, pointsInCluster, assignment]=
kmeans(data, nbCluster)

Output:
centroid: matrix in each row are the Coordinates of a centroid
pointsInCluster: row vector with the nbDatapoints belonging to
the centroid
assignment: row Vector with clusterAssignment of the dataRows

Input:
data in rows
nbCluster : nb of centroids to determine

(c) by Christian Herta ( www.christianherta.de )

'''
def kmeans(data, nbCluster):
    data_dim = len(data[0])
    nbData = len(data)
    
    # init the centroids randomly
    data_min = [min(data[:,0]), min(data[:,1]), min(data[:,2])]
    data_max = [max(data[:,0]), max(data[:,1]), max(data[:,2])]
    
    data_diff = numpy.subtract(data_max, data_min)

    # every row is a centroid
    centroid = numpy.random.rand(nbCluster, data_dim)
    
    x = centroid[0,:] * data_diff
    for i in range(len(centroid[:,1])):
        centroid [i,:] = centroid[i,:] * data_diff
        centroid [i,:] = centroid[i,:] + data_min

    # end init centroids
    
    # no stopping at start
    pos_diff = 1.0
    # main loop

    while pos_diff > 0.1:
        
        # E-step
        assignment = []
        
        # assign each datapoint to the closest centroid
        for d in range(len(data[:,0])):
            min_diff = numpy.subtract(data[d,:], centroid[0,:])
            min_diff = numpy.dot(min_diff, min_diff.transpose())
            curAssignment = 0
            for c in range(1, nbCluster):
                diff2c = numpy.subtract(data[d,:], centroid[c,:])
                diff2c = numpy.dot(diff2c, diff2c.transpose())
                if min_diff >= diff2c:
                    curAssignment = c
                    min_diff = diff2c
            
            # assign the d-th dataPoint
            assignment.append(curAssignment)
            # for the stoppingCriterion
        oldPositions = centroid
            
        # M-Step
        # recalculate the positions of the centroids
        
        centroid = numpy.zeros(shape = (nbCluster, data_dim))
        pointsInCluster = numpy.zeros(shape = (nbCluster, 1))
        for d in range(len(assignment)):
            centroid[assignment[d],:] = centroid[assignment[d],:] + data[d,:]
            pointsInCluster[assignment[d], 0] = pointsInCluster[assignment[d], 0] + 1
        
        for c in range(nbCluster):
            
            if pointsInCluster[c, 0] != 0:
                centroid[c, :] = centroid[c, :] / pointsInCluster[c, 0]
            else:
                # set cluster randomly to new position
                centroid[c, :] = (numpy.random.rand(1, data_dim) * data_diff) + data_min
            
            # stoppingCriterion
#             print("Centroids: ", centroid, "\nOld:", oldPositions)
        pos_diff = sum(sum((centroid - oldPositions) **2))
#         print(pos_diff)
    return (centroid, pointsInCluster, assignment)
    
def mat_div(m, x):
    z = numpy.zeros((len(m),len(m[0])))
    for i in range(len(m)):
        for j in range(len(m[0])):
            z[i][j] = m[i][j] / x
    return z 
    
def get_xyz_data (path, name):
    fx = open(path + os.sep + name + '_x.csv', 'r')
    fy = open(path + os.sep + name + '_y.csv', 'r')
    fz = open(path + os.sep + name + '_z.csv', 'r')
    
    cx = csv.reader (fx, delimiter = ',', quotechar='|')
    cy = csv.reader (fy, delimiter = ',', quotechar='|')
    cz = csv.reader (fz, delimiter = ',', quotechar='|')
    
    x = get_float_matrix(cx)
    y = get_float_matrix(cy)
    z = get_float_matrix(cz)
    
    fx.close()
    fy.close()
    fz.close()
    
    m = []
    m.append(x)
    m.append(y)
    m.append(z)

    return m
    
def get_float_matrix(c):
    
    m = []
    for row in c:
        r = []
        for item in row:
            r.append(float(item))
        m.append(r)
    
    return m


if __name__ == '__main__':
    main ()