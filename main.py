'''
Created on Jun 6, 2013

@author: Amin
'''

import csv
import os
import numpy
import math
import sys
from scipy import sparse

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
    testing = get_xyz_data(path + 'test',test_gesture);
    
            
    '''
    ****************************************************
    *  Initializing
    ****************************************************
    '''
    gestureRecThreshold = 0 # set below
    
    centroids, NX = get_point_centroids(training,N,D)
    
#     print(len(training[0][0][:]))
    ATrainBinned = get_point_clusters(training,centroids,D)
    ATestBinned = get_point_clusters(testing,centroids,D)
    
    '''
    ****************************************************
    *  Training
    ****************************************************
    '''
    
    # Set priors
    pP = prior_transition_matrix(M,LR)
    
    # Train the model:
    cyc = 50
    dhmm_numeric(ATrainBinned,pP,range(N),M,cyc,.00001)
    
def dhmm_numeric(X, pP, bins, K = 2, cyc = 100, tol = 0.0001):
    
    D = len(X[0,0,:])
    
    num_bins = len(bins)
    epsilon = sys.float_info.epsilon
    
    # number of sequences
    N = len(X[:,0,0])
    
    # calculating the length of each sequence
    T = numpy.ones(shape = (1, N))
    
    for n in range(N):
        T[0, n] = len(X[0, :, 0])
    
    TMAX = T.max()
    
#     print('\n********************************************************************\n');
#     print('Training %d sequences of maximum length %d from an alphabet of size %d\n', N, TMAX, num_bins);
#     print('HMM with %d hidden states\n',K);
#     print('********************************************************************\n');

    E = 0.1 * numpy.random.rand(num_bins, K)
    E = E + numpy.ones(shape = (num_bins, K))
    E = E / num_bins
    E = cDiv(E, cSum(E))
    
    B = numpy.zeros(shape = (TMAX, K))
    
    Pi = numpy.random.rand(K)
    Pi = Pi / sum(Pi)
    
    # transition matrix
    P = pP
    P = rDiv(P, rSum(P))
    
    P = sparse.lil_matrix(P)
    Xcurrent = X[1, :, :]

    
    LL = []
    lik = 0
#     for cycle in range(cyc):
    GammaInit = numpy.zeros(shape = (1, K))
    GammaSum = numpy.zeros(shape = (1, K))
    GammaKSum = numpy.zeros(shape = (num_bins, K))
    Scale = numpy.zeros(shape = (TMAX, 1))
    sxi = sparse.lil_matrix((K, K))
        
    for n in range(N):
        alpha = numpy.zeros(shape = (int(T[0,n]), K))
        beta = numpy.zeros(shape = (int(T[0,n]), K))
        gamma = numpy.zeros(shape = (int(T[0,n]), K))
        gammaKsum = numpy.zeros(shape = (num_bins, K)) # not GammaKSum!
        
        #Inital values of B = Prob(output|s_i), given data X
        n = 0
        Xcurrent = X[n, :, :]
        #print(Xcurrent[1] == bins)
        for i in range(int(T[0,n])):
            crit = (Xcurrent[i] == bins)
            m = numpy.where(crit)[0][0]
            if sum(crit) < 1:
                print("Error: Symbol not found")
                return
            B[i, :] = E[m, :]
        
        scale = numpy.zeros(shape = (int(T[0,n]),1)) # NOT Scale
        alpha[0, :] = Pi.transpose() * B[0,:]
        scale[0,0] = sum(alpha[0,:])
        alpha[0,:] = alpha[0,:]/scale[0,0]
        
        for i in range(1,int(T[0,n])):
            alpha[i,:] = (alpha[i-1,:]*P) * B[i,:]
            scale[i,0] = sum(alpha[i,:])
            alpha[i,:] = alpha[i,:]/scale[i,0]
        
        beta[int(T[0,n]) - 1,:] = numpy.ones(shape = (1, K)) / scale [int(T[0,n]) - 1,0]
            
        for i in range(int(T[0,n]) - 2, -1, -1): #Starting from the second last index and counting down to 0
            beta[i,:] = (beta[i + 1,:] * B[i + 1,:]) * P.transpose()/scale[i,0]
            
        gamma = (alpha * beta) + epsilon
        gamma = rDiv(gamma,rSum(gamma))
        
        gammaSum = sum (gamma)
        
        for i in range(int(T[0,n])):
            # find the letter in the alphabet
            crit = (Xcurrent[i] == bins)
            m = numpy.where(crit)[0][0]
            gammaKsum[m, :] = gammaKsum[m, :] + gamma [i, :]
        
     
        for i in range(int(T[0, n]) - 1):
            alphaTrans = numpy.zeros(shape = (1,len(alpha[i,:])))
            alphaTrans[0,:] = alpha[i,:].transpose()
            
            betaMB = numpy.zeros(shape = (len(beta[i + 1, :]), 1))
            betaMB[:,0] = beta[i + 1, :]* B[i + 1, :]
            
            fin = numpy.dot(betaMB,alphaTrans)
            t = P.multiply(fin)
            sxi = sxi + t / numpy.sum(t)
        sxi = sparse.lil_matrix(sxi)
            
        GammaInit = GammaInit + gamma[0,:]
        GammaSum = GammaSum + gammaSum
        
        for i in range(int(T[0, n]) - 1):
            Scale[i,:] = Scale[i,:] + numpy.log(scale[i,:])
        
        Scale[int(T[0, n]) - 1,:] = Scale[int(T[0, n]) - 1,:] + numpy.log(scale[int(T[0, n]) - 1,:])
        
    """
    NOTE!
    Next is to adapt the M step and integrate into the cycles
    """
    print(Scale)
    pass

def rDiv (X, Y):
    N, M = X.shape
    S = Y.shape
    if len(S) > 1:
        K, L = S
    else:
        K = S[0]
        L = 1
    
    if N!=K or L!=1:
        print("Error in Row division!")
        return None
            
    Z = numpy.zeros_like(X)
    
    if M < N:
        for m in range(M):
            Z[:, m] = X[:, m] / Y
    else:
        for n in range(N):
            Z[n, :] = X [n, :] / Y[n]
    
    return (Z)

def rSum(X):
    N, M = X.shape
    
    Z = numpy.zeros(N)
    
    if M == 1:
        Z = X
    elif M < 2 * N:
        for m in range(M):
            Z = Z + X[:, m]
    else:
        for n in range(N):
            Z[n] = numpy.sum(X[n, :])
    return (Z)


def cSum (X):
    '''
    Column sum
    '''
    if len(X[:, 0]) > 1:
        Z = numpy.sum(X, axis = 0)
    else:
        Z = X
        
    return (Z)

def cDiv(X, Y):
    if (len(X[0, :]) != len(Y)) or (len(Y.shape) != 1):
        print('Error in Column  Division')
        return (None)
     
    Z = numpy.zeros_like(X)
        
    for i in range(len(X[:,0])):
        Z[i, :] = X[i, :] / Y
    
    return (Z)

def prior_transition_matrix(K, LR):
    '''
    ****************************************************
    *  Create a prior for the transition matrix
    ****************************************************
    '''
    
    #LR is the allowable number of left-to-right transitions
    P = (1 / LR) * numpy.identity(K)
    
    for i in range(K - (LR - 1)):
        for j in range((LR - 1)):
            P[i, i + (j + 1)] = 1 / LR

    for i in range(K - (LR - 1), K):
        for j in range(K - i):
            P[i, i + j] = 1 / ((K - 1) - i + 1)
    return (P)

def get_point_clusters(data,centroids,D):
    XClustered = numpy.zeros(shape = (len(data[0][0][:]), len(data[0][:][:]), 1))
    K = len(centroids[:,0])
    

    for n in range(len(data[0][:][:])):
        for i in range(len(data[0][0][:])):
            temp = numpy.zeros(shape = (K, 1))
            for j in range(K):
                if D == 3:
                    temp[j] = math.sqrt((centroids[j,0] - data[0][n][i])**2+(centroids[j,1] - data[1][n][i])**2+(centroids[j,2] - data[2][n][i])**2)
            
            idx, I = min((val, idx) for (idx, val) in enumerate(temp))
            XClustered[i,n,0] = I # TO FINISH AND RETURN CORRECTLY
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
    return (centroids, K)


def kmeans(data, nbCluster):
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