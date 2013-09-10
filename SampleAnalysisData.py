import csv
import numpy
import os
import random
import classifier


D = 3  # the number of dimensions to use: X, Y, Z
M = 12  # output symbols
N = 8  # states
LR = 2  # degree of play in the left-to-right HMM transition matrix 

def main ():
    path = 'data\\observations\\training'
    name = '1'
    axis = 2
    
    trs = get_all(path + os.sep + 'flying\\hand_l\\down')
    
    #print(trs["d1"])
    #extended = extend_list(trs["d1"], 16, 2)
    #print(extended)
    #return
    ds = []
    us = []
    maxLenD = maxLenU = 0
    for i in range(len(trs)):
        if i % 2 == 0:
            key = "d" + str(int(i/2) + 1)
            d = trs[key]
            #print(key, d[:,axis].min(), d[:,axis].max(), len(d))
            #print(extend_list(d, 16, 2))
            ds.append(d)
            if len(d) > maxLenD:
                maxLenD = len(d)
        else:
            key = "u" + str(int(i/2) + 1)
            d = trs[key]
            #print(key, d[:,axis].max(), d[:,axis].min(), len(d))
            us.append(d)
            if len(d) > maxLenU:
                maxLenU = len(d)
    #return
    allD = unified_len(ds, maxLenD, D)
    allU = unified_len(us, maxLenD, D)

    TRTS = classifier.classifier()
    centroids = TRTS.get_point_centroids(allD, N, D)
    #print(centroids)
    #print(ds)
    #print(us)
    tst = extend_list(trs["d1"], maxLenD, 2)
    testing = numpy.zeros(shape = (maxLenD, 2, 3))
    testing[:,0,:] = tst[:,:]
    tst = extend_list(trs["u1"], maxLenD, 2)
    testing[:,1,:] = tst[:,:]
    #print(testing.shape)
    #return
    ATrainBinned = TRTS.get_point_clusters(allD, centroids, D)
    ATestBinned = TRTS.get_point_clusters(testing, centroids, D)
    #print(ATestBinned.shape)
    #return
    '''
    ****************************************************
    *  Training
    ****************************************************
    '''
    
    # Set priors
    pP = TRTS.prior_transition_matrix(M, LR)
    
    # Train the model:
    b = [x for x in range(N)]
    cyc = 50
    E, P, Pi, LL = TRTS.dhmm_numeric(ATrainBinned, pP, b, M, cyc, .00001)
    
    '''
    ****************************************************
    *  Testing
    ****************************************************
    '''
    
    sumLik = 0
    minLik = numpy.Infinity
    
    
    for j in range(len(ATrainBinned)):
        lik = TRTS.pr_hmm(ATrainBinned[j], P, E.transpose(), Pi)
        if lik < minLik:
            minLik = lik
        sumLik = sumLik + lik
    
    gestureRecThreshold = 2.0 * sumLik / len(ATrainBinned)

    
    
    print('\n********************************************************************')
    print('Testing %d sequences for a log likelihood greater than %.4f' % (len(ATestBinned), gestureRecThreshold))
    print('********************************************************************\n');
    
    gesture = "Down"
    recs = 0
    tLL = numpy.zeros(shape=(len(ATestBinned)))
    for j in range(len(ATestBinned)):
        tLL[j] = TRTS.pr_hmm(ATestBinned[j], P, E.transpose(), Pi)
        if tLL[j] > gestureRecThreshold:
            recs = recs + 1
            print("Log Likelihood: %.3f > %.3f (threshold) -- FOUND %s Gesture" % (tLL[j], gestureRecThreshold, gesture))
        else:
            print("Log Likelihood: %.3f < %.3f (threshold) -- NO %s Gesture" % (tLL[j], gestureRecThreshold, gesture))
        
    print('Recognition success rate: %.2f percent\n' % (100 * recs / len(ATestBinned)))
        


    
def unified_len(data, l, D):
    unified = numpy.zeros(shape = (l, len(data), D))
    for i in range(len(data)):
        unified[:, i, :] = extend_list(data[i], l, 2)
    
    return unified 
    
def extend_list(data, s, axis):
    if len(data) > s:
        raise Exception("Size is greater than the length.")
    elif len(data) == s:
        return data
    else:
        max_distance = 0
        index = 0
        for i in range(len(data) - 1):
            distance = abs(data[i + 1, axis] - data[i, axis])
            if distance > max_distance:
                max_distance = distance
                index = i
        one_extended = numpy.zeros(shape = (len(data) + 1, len(data[0,:])))
        one_extended[0:index + 1,:] = data[0:index + 1,:]
        one_extended[index + 1,:] = (data[index,:] + data[index + 1,:]) / 2
        one_extended[index + 2:, :] = data[index + 1:, :]
        
        return extend_list(one_extended, s, axis)
        
def get_all (path):
    d = {}
    files = os.listdir(path)
    for file in files:
        if os.path.isfile(path + os.sep + file):
            fName = file[0:len(file) - 4]
            d[fName] = get_csv_format(path + os.sep + file)
    
    return d

def get_csv_format(path):
    f = open(path, "r")
    c = csv.reader (f, delimiter=',', quotechar='|')
    x = numpy.asarray(list(c), dtype = 'float')
    return x


    
if __name__ == '__main__':
    main()