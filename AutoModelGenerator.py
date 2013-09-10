import csv
import numpy
import os
import classifier
import sys

gestures = ['breaststroke', 'crawl', 'dogpaddle', 'flying'] #['waving']
limbs = ['elbow_l', 'elbow_r', 'hand_l', 'hand_r']
dirs = ['down', 'up'] #['left', 'right']

D = 3  # the number of dimensions to use: X, Y, Z
LR = 2  # degree of play in the left-to-right HMM transition matrix

M = 6  # output symbols
N = 4  # states

def main():
    axis = 2
    gest_no = 2
    limb_no = 1 
    dir_no = 0
    models_count = 0
    #path = 'data\\observations\\training\\' + 'circle' #+ gest + os.sep + limb + os.sep + dir
    #model_name = 'circle' #gest + '_' + limb + '_' + dir
    #train_store(path, 'data\\observations\\', axis, model_name)
    #return
    for gest in gestures:
        for limb in limbs:
            for dir in dirs:
                print('Generating the model for %s of %s in %s...' %(gest, limb, dir))
                path = 'data\\observations\\training\\' + gest + os.sep + limb + os.sep + dir
                model_name = gest + '_' + limb + '_' + dir
                train_store(path, 'data\\observations\\', axis, model_name)
                models_count = models_count + 1
    print('\n\n\n%d models are created and stored' % models_count)
    return
    

def train_store(path, path_mode, axis, model_name):
    
    trs, maxLen = get_all(path)
    training = unified_len(trs, maxLen, D, axis)
    TRTS = classifier.classifier()
    centroids = TRTS.get_point_centroids(training, N, D)
    
    ATrainBinned = TRTS.get_point_clusters(training, centroids, D)
    
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
    
    sumLik = 0
    minLik = numpy.Infinity
    
    for j in range(len(ATrainBinned)):
        lik = TRTS.pr_hmm(ATrainBinned[j], P, E.transpose(), Pi)
        if lik < minLik:
            minLik = lik
        sumLik = sumLik + lik
    
    gestureRecThreshold = 2.0 * sumLik / len(ATrainBinned)
    print("The threshold is ", gestureRecThreshold)
    store_model(path_mode, model_name, E, P, Pi, centroids, gestureRecThreshold)
    


def store_model(path, name, E, P, Pi, cent, thr):
    directory = "model"
    if not os.path.exists(path + os.sep + directory):
        os.makedirs(path + os.sep + directory)

    Ef = open(path + os.sep + "model" + os.sep + name + ".csv", "w")
    Ewriter = csv.writer(Ef, delimiter = ',', quotechar = '', quoting = csv.QUOTE_NONE, dialect = csv.unix_dialect)
    Ewriter.writerow(E.shape)
    Ewriter.writerows(E)
    Ewriter.writerows(P)
    Ewriter.writerow(Pi)
    Ewriter.writerows(cent)
    Ewriter.writerow([thr])
    Ef.close()
    print("Model " + name + " is created at " + path)
    
def unified_len(data, l, D, axis):
    unified = numpy.zeros(shape = (l, len(data), D))
    iter = 0
    for tr in data:
        cur = data[tr]
        extended = extend_list(cur, l, axis)
        unified[:, iter, :] = extended
        iter = iter + 1
    
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
    maxLen = 0
    for file in files:
        if os.path.isfile(path + os.sep + file):
            fName = file[0:len(file) - 4]
            d[fName] = get_csv_format(path + os.sep + file)
            currentLen = len(d[fName])
            if currentLen > maxLen:
                maxLen = currentLen
    
    return (d, maxLen)

def get_csv_format(path):
    f = open(path, "r")
    c = csv.reader (f, delimiter=',', quotechar='|')
    x = numpy.asarray(list(c), dtype = 'float')
    x = x - x[0,:]
    return x


if __name__ == '__main__':
    for arg in sys.argv:
        pass
        
    main()