'''
Created on Jul 20, 2013

@author: Amin
'''
import dataflow
import classifier
import numpy
import os
import csv
from modelLoader import Loader

gesture = 'x'
D = 3
def main():
    
    path = "data\\model\\"
    modelLoader = Loader()
    models = modelLoader.getAllModels(path)
    
    files = modelLoader.getAllFileNames("data\\tests")
    names = [f.partition(".csv")[0] for f in files]
    tests = modelLoader.loadTest("data\\tests\\", files)
#     data = dataflow.dataflow("data", gesture)

    TS = classifier.classifier()
#     E, P, Pi, cent, gestureRecThreshold = data.load_model()
    for _, testName in enumerate(tests):
        oneTest = numpy.empty(shape = (60, 1, 3))
        oneTest[:,0,:] = tests[testName]
        for j, gesture in enumerate(models):
            model = models[gesture]
            E, P, Pi, cent, gestureRecThreshold = (model["E"], model["P"], model["Pi"], model["centroids"], model["threshold"]) 
            ATestBinned = TS.get_point_clusters(oneTest, cent, D)
            tLL = TS.pr_hmm(ATestBinned[0], P, E.transpose(), Pi)
            if tLL > gestureRecThreshold:
                print("Log Likelihood: %.3f > %.3f (threshold) -- FOUND %s Gesture" % (tLL, gestureRecThreshold, gesture))
            else:
                print("Log Likelihood: %.3f < %.3f (threshold) -- NO %s Gesture" % (tLL, gestureRecThreshold, gesture))
            
        
 
    return
#     print(i)
    oneTest = tests["1"]
    #oneTest = data.get_tests_attached(oneTest)
    ATestBinned = TS.get_point_clusters(oneTest, cent, D)
    print(ATestBinned.shape)
#     print('\n********************************************************************')
#     print('Testing %d sequences for a log likelihood greater than %.4f' % (len(ATestBinned), gestureRecThreshold))
#     print('********************************************************************\n');
    
    recs = 0
    tLL = numpy.zeros(shape=(len(ATestBinned), 1))
    
    for j in range(len(ATestBinned)):
        tLL[j, 0] = TS.pr_hmm(ATestBinned[j], P, E.transpose(), Pi)
        if tLL[j, 0] > gestureRecThreshold:
            recs = recs + 1
            print("Log Likelihood: %.3f > %.3f (threshold) -- FOUND %s Gesture" % (tLL[j, 0], gestureRecThreshold, gesture))
        else:
            print("Log Likelihood: %.3f < %.3f (threshold) -- NO %s Gesture" % (tLL[j, 0], gestureRecThreshold, gesture))
        
    print('Recognition success rate: %.2f percent\n' % (100 * recs / len(ATestBinned)))
    
    
#     Ef = open("data" + os.sep + "model" + os.sep + gesture + "1.csv", "w")
#     Ewriter = csv.writer(Ef, delimiter = ',', quotechar = '', quoting = csv.QUOTE_NONE, dialect = csv.unix_dialect)
#     Ewriter.writerows(oneTest[:,0,:])
#     Ef.close()

def sepTests():
    gestures = ["circle", "l", "m", "round", "x", "z"]
    count = 0
    for gesture in gestures:
        data = dataflow.dataflow("data", gesture)
        test = data.get_test_xyz()
        for i in range(test.shape[1]):
            count = count + 1
            Ef = open("data\\tests\\" + str(count) + ".csv", "w")
            Ewriter = csv.writer(Ef, delimiter = ',', quotechar = '', quoting = csv.QUOTE_NONE, dialect = csv.unix_dialect)
            Ewriter.writerows(test[:,i,:])
            Ef.close()
            
        print(test.shape)

def fun(*args):
    print((args[0]))
    for arg in args:
        print(arg) 
        
if __name__ == '__main__':
    main()