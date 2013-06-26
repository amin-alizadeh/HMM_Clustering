'''
Created on Jun 6, 2013

@author: Amin
'''

import csv
import os
import numpy

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
    path = 'practice' + os.sep 
    training = get_xyz_data(path + 'train', train_gesture);
    testing = get_xyz_data(path + 'test',test_gesture);
    
    #print(training)
    #print(len(training))
    print(training[0][0][0])
    print(testing[0][0][0])
    numpy.zeros (shape = (2,3))
    
    '''
    ****************************************************
    Initializing
    ****************************************************
    '''
    gestureRecThreshold = 0 # set below
    
    centroids, N = get_point_centroids(training,N,D)
    
     
def get_point_centroids(data,K,D):
    
    mean = numpy.zeros(shape = (len(data[0]),D))
    
    
    pass

    
    
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