import os
import csv
import numpy
import classifier

D = 3  # the number of dimensions to use: X, Y, Z
M = 12  # output symbols
N = 8  # states
LR = 2  # degree of play in the left-to-right HMM transition matrix 

def main():
    path = "data\\Coords\\training"
    training = get_all_training(path)
    testing = get_all_training("data\\Coords\\testing")
    TRTS = classifier.classifier()
    
    centroids = TRTS.get_point_centroids(training, N, D)
    diff_test = transformation(testing, training)
    testing = diff_test
    ATrainBinned = TRTS.get_point_clusters(training, centroids, D)
    ATestBinned = TRTS.get_point_clusters(testing, centroids, D)
    
    print(ATrainBinned, ATestBinned)
    
    pP = TRTS.prior_transition_matrix(M, LR)
    
    # Train the model:
    b = [x for x in range(N)]
    cyc = 50
    E, P, Pi, LL = TRTS.dhmm_numeric(ATrainBinned, pP, b, M, cyc, .00001)
    
    
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
    
    gesture = "Gesture"
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
        
def transformation(data, dataTo):
#     print("Centroids: ", dataTo)
    
    tr_diff = numpy.empty_like(data)
    for i in range(len(data[0,:,0])):
        diff = data[0,i,:] - dataTo[0,0,:]
        print("Diff ", diff)
        tr_diff[:,i,:] = data[:,i,:] - diff
    print("Data: ", data)
    print("New: ", tr_diff)
    return tr_diff
    
def get_all_training(path):
    fs = get_all_filenames(path)
    
    allT = []
    maxLenT = 0
    for f in fs:
        ft = open(path + os.sep + f)
        ct = csv.reader (ft, delimiter=',', quotechar='|')
        t = numpy.asarray(list(ct), dtype = 'float')
        allT.append(t)
        if len(t) > maxLenT:
            maxLenT = len(t)
    
    tr = numpy.empty(shape = (maxLenT, len(fs), 3))
    for i in range(len(allT)):
        tr[:,i,:] = allT[i]
    return tr
         
         
def get_all_filenames(path):
    files = []
    for file in os.listdir(path):
        files.append(file)
    return files


if __name__ == '__main__':
    main()