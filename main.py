'''
Created on Jun 6, 2013

@author: Amin
'''

import numpy
import math
import dataflow
import classifier

D = 3  # the number of dimensions to use: X, Y, Z
M = 12  # output symbols
N = 8  # states
LR = 2  # degree of play in the left-to-right HMM transition matrix 

gesture = 'z'


def main ():

    data = dataflow.dataflow("data", gesture)
    training = data.get_train_xyz()
    testing = data.get_test_xyz()
    '''
    ****************************************************
    *  Initializing
    ****************************************************
    '''
    TRTS = classifier.classifier()
    
    centroids = TRTS.get_point_centroids(training, N, D)
    ATrainBinned = TRTS.get_point_clusters(training, centroids, D)
    ATestBinned = TRTS.get_point_clusters(testing, centroids, D)
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
    
    ATrainBinned = ATrainBinned[:,:,0]
    ATestBinned = ATestBinned[:,:,0]
    
    for j in range(len(ATrainBinned)):
        lik = TRTS.pr_hmm(ATrainBinned[j], P, E.transpose(), Pi)
        if lik < minLik:
            minLik = lik
        sumLik = sumLik + lik
    
    gestureRecThreshold = 2.0 * sumLik / len(ATrainBinned)

    data.store_model(E, P, Pi, centroids, gestureRecThreshold)
#     data.store_Binned(ATrainBinned[:,:,0],ATestBinned[:,:,0])
    
    
    print('\n********************************************************************')
    print('Testing %d sequences for a log likelihood greater than %.4f' % (len(ATestBinned), gestureRecThreshold))
    print('********************************************************************\n');
    
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
        
    
if __name__ == '__main__':
    main ()
