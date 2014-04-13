import os
import sys
import csv
import numpy
import math

centroids_path = "all-centroids.csv"
root_path = "data" + os.sep + "unitytrain"
gesture_name = "circle-r-ccw"
orient = "R"
joint_header = "Hand_" + orient
parent_joint_header = "Shoulder_" + orient
D = 3  # the number of dimensions to use: X, Y, Z
M = 12  # output symbols
N = 8  # states
LR = 2  # degree of play in the left-to-right HMM transition matrix 

def main ():
    
    training_path = root_path + os.sep + gesture_name
    allFiles = get_All_Files (training_path)
    all_trains = [] 
    """
        Array of dictionaries. Each dictionary contains coordinates of  
        the joints. The name of the joints are the keys of the dictionaries
    """
    for file in allFiles:
        all_trains.append(get_xyz_data(file))
        
    """
    joints:
    Dictionary: keys -> joints such as "Shoulder_R", "Hand_R", etc.
                values -> List of all coordinates
                        elements of the list -> Numpy array of the coordinates
    """
   
    normalized_trains = normalize_joint_with_parent (all_trains, joint_header, parent_joint_header)
    centroids = get_centroids_data(root_path + os.sep + centroids_path)
    
    train_binned = get_point_clusters(normalized_trains, centroids)
    
    '''
    ****************************************************
    *  Training
    ****************************************************
    '''
    
    # Set priors
    pP = prior_transition_matrix(M, LR)
    
     # Train the model:
    b = [x for x in range(len(centroids))]
    cyc = 60
    E, P, Pi, LL = dhmm_numeric(train_binned, pP, b, M, cyc, .00001)
    
    sumLik = 0
    minLik = numpy.Infinity

    for i in range(len(train_binned)):
        lik = pr_hmm(train_binned[i], P, E, Pi)
        if lik < minLik:
            minLik = lik
        sumLik = sumLik + lik
    
    gestureRecThreshold = 2.0 * sumLik / len(train_binned)
    store_model(E, P, Pi, gestureRecThreshold, root_path, gesture_name)
    print("Model %s with threshold %.3f saved successfully..." %(gesture_name, gestureRecThreshold))

def store_model(E, P, Pi, thr, path, name):
    Ef = open(path + os.sep + "model" + os.sep + name + ".csv", "w")
    Ewriter = csv.writer(Ef, delimiter = ',', quotechar = '', quoting = csv.QUOTE_NONE, dialect = csv.unix_dialect)
    Ewriter.writerow(E.shape)
    Ewriter.writerows(E)
    Ewriter.writerows(P)
    Ewriter.writerow(Pi)
    Ewriter.writerow([thr])
    Ef.close()
    
def pr_hmm(clusters, P, E, pi):
    n = P.shape[1] # Which is M, the number of hidden symbols
    T = len(clusters) # The number of clustered observed data
    m = numpy.zeros(shape=(T, n))
    # it uses forward algorithm to compute the probability
    for i in range(n):  # initilization
        m[0, i] = E[int(clusters[0]), i] * pi[i]
        
    for t in range(T - 1):  # recursion
        for j in range(n):
            z = 0
            for i in range(n):
                z = z + P[i, j] * m[t, i]
            m[t + 1, j] = z * E[int(clusters[t + 1]), j]
        
    p = 0
    
    for i in range(n):  # termination
        p = p + m[T - 1, i]
    
    if p != 0:
        p = math.log(p)
    else:
        p = float('-inf')
        
    return p

def prior_transition_matrix(K, LR):
    '''
    ****************************************************
    *  Create a prior for the transition matrix
    ****************************************************
    '''
    
    # LR is the allowable number of left-to-right transitions
    P = (1 / LR) * numpy.identity(K)
    
    for i in range(K - (LR - 1)):
        for j in range((LR - 1)):
            P[i, i + (j + 1)] = 1 / LR

    for i in range(K - (LR - 1), K):
        for j in range(K - i):
            P[i, i + j] = 1 / ((K - 1) - i + 1)
    return (P)   

def get_point_clusters(all_data, centroids):
    all_clustered = []
    number_of_clusters = len(centroids)
    for data in all_data:
        length = len(data)
        XClustered = numpy.empty(length)
        for i in range(length):
            temp = numpy.zeros(shape=(number_of_clusters))
            for j in range(number_of_clusters):
                temp[j] = math.sqrt((centroids[j, 0] - data[i, 0]) ** 2 + \
                                            (centroids[j, 1] - data[i, 1]) ** 2 + \
                                            (centroids[j, 2] - data[i, 2]) ** 2)
        
            idx, I = min((val, idx) for (idx, val) in enumerate(temp))
            XClustered[i] = I
        all_clustered.append(XClustered)
        
    return all_clustered 
    all_clustered = {}
    headers = list(all_data.keys())
    for header in headers:
        clustered = []
        for data in all_data[header]:
            length = len(data)
            XClustered = numpy.empty(length)
            centroids = all_centroids[header]
            number_of_clusters = len(centroids)
            
            for i in range(length):
                temp = numpy.zeros(shape=(number_of_clusters))
                for j in range(number_of_clusters):
                    temp[j] = math.sqrt((centroids[j, 0] - data[i, 0]) ** 2 + \
                                        (centroids[j, 1] - data[i, 1]) ** 2 + \
                                        (centroids[j, 2] - data[i, 2]) ** 2)
                
                idx, I = min((val, idx) for (idx, val) in enumerate(temp))
                XClustered[i] = I
            clustered.append(XClustered)
            

        all_clustered[header] = clustered
            
    
    return all_clustered
def get_centroids_data(path):
    f = open(path)
    c = csv.reader (f, delimiter=',', quotechar='|')
    d = numpy.asarray(list(c), dtype = 'float')
    f.close()
    return d
 
def normalize_joint_with_parent (data, header, parent):
    normalized_header = []
    for i in range(len(data)):
        normalized = data[i][header] - data[i][parent]
        normalized_header.append(normalized)
    
    return normalized_header
   
def get_All_Files (path):
    files = []
    for file in os.listdir(path):
        files.append(path + os.sep + file)
    return files

def get_xyz_data(path):
    f = open(path)
    c = csv.reader (f, delimiter=',', quotechar='|')
    first_row = next(f, None).split(',')
    headers = [first_row[h] for h in range(0, len(first_row), 3)]

    m = numpy.asarray(list(c), dtype = 'float')
    d = {}
    for header, i in zip(headers, range(len(headers))):
        d[header] = m[:, (i * 3):((i + 1) * 3)]
    f.close()
    return d

def get_xyz_data_no_header(path):
    f = open(path)
    c = csv.reader (f, delimiter=',', quotechar='|')
    d = numpy.asarray(list(c), dtype = 'float')
    f.close()
    return d


def rDiv (X, Y):
    N, M = X.shape
    S = Y.shape
    if len(S) > 1:
        K, L = S
    else:
        K = S[0]
        L = 1
    
    if N != K or L != 1:
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
        Z = numpy.zeros(shape=(N))
        if M == 1:
            Z = X
        elif M < 2 * N:
            for m in range(M):
                Z = Z + X[:, m]
                # FIX THE SHAPE
        else:
            for n in range(N):
                Z[0, n] = numpy.sum(X[n, :])
        return (Z) 
 
def cSum (X):
    '''
       Column sum
    '''
    Z = numpy.zeros(shape=(1, len(X[0, :])))
    if len(X[:, 0]) > 1:
        Z[0, :] = numpy.sum(X, axis=0)
    else:
        Z[0, :] = X
        
    return (Z)


def cDiv(X, Y):
    if (X.shape[1] != Y.shape[1]):
        print('Error in Column Division shapes')
        return (None)
    elif len(Y.shape) > 1:
        if Y.shape[0] != 1:
            print('Error in Column Division')
            return (None)
            
     
    Z = numpy.zeros_like(X)
    
    for i in range(len(X[:, 0])):
        Z[i, :] = X[i, :] / Y
    
    return (Z)

    
def dhmm_numeric(data, pP, bins, K=12, cyc=100, tol=0.0001):
    
    num_bins = len(bins)
    epsilon = sys.float_info.epsilon
    # number of sequences
    N = len(data)
    
    
    # calculating the length of each sequence
    T = numpy.empty(shape=(1, N))
    
    for n in range(N):
        T[0, n] = len(data[n])
    
    TMAX = T.max()
    
    
    print('********************************************************************');
    print('Training %d sequences of maximum length %d from an alphabet of size %d' % (N, TMAX, num_bins));
    print('HMM with %d hidden states' % K);
    print('********************************************************************');
    
    rd = numpy.random.rand(num_bins, K)

    E = 0.1 * rd
    E = E + numpy.ones(shape=(num_bins, K))
    E = E / num_bins
    E = cDiv(E, cSum(E))

    B = numpy.zeros(shape=(TMAX, K))
    
    Pi = numpy.random.rand(K)
    
    #REMOVE THE FOLLOWING
    #Pi = numpy.array([ 0.94503978,  0.97160498,  0.54553485,  0.34016857,  0.48067553,  0.94473749,  0.57018499,  0.38132459,  0.40252906,  0.61204666,  0.21091647,  0.0745824 ])
    
    Pi = Pi / numpy.sum(Pi)

    # transition matrix
    P = pP
    P = rDiv(P, rSum(P))
    
    #P = sparse.lil_matrix(P)

    LL = []
    lik = 0

    for cycle in range(cyc):
        # FORWARD-BACKWARD
        Gammainit = numpy.zeros(shape=(1, K))
        Gammasum = numpy.zeros(shape=(1, K))
        Gammaksum = numpy.zeros(shape=(num_bins, K))
        Scale = numpy.zeros(shape=(TMAX, 1))
        sxi = numpy.zeros(shape = (K, K)) #sparse.lil_matrix(K, K)
        
        for n in range(N):
            alpha = numpy.zeros(shape=(int(T[0, n]), K))
            beta = numpy.zeros(shape=(int(T[0, n]), K))
            gamma = numpy.zeros(shape=(int(T[0, n]), K))
            gammaksum = numpy.zeros(shape=(num_bins, K))  # not Gammaksum!
            
            # Inital values of B = Prob(output|s_i), given data data
            Xcurrent = data[n]

            for i in range(int(T[0, n])):
                crit = (Xcurrent[i] == bins)
                m = numpy.where(crit)[0][0]
                if sum(crit) < 1:
                    print("Error: Symbol not found")
                    return
                B[i, :] = E[m, :]
            
            scale = numpy.zeros(shape=(int(T[0, n]), 1))  # NOT Scale
            alpha[0, :] = Pi.transpose() * B[0, :] 
            scale[0, 0] = numpy.sum(alpha[0, :])
            alpha[0, :] = alpha[0, :] / scale[0, 0]
                
            
            for i in range(1, int(T[0, n])):
                #alpha[i, :] = (numpy.dot(alpha[i - 1, :], P.toarray())) * B[i, :]
                alpha[i, :] = numpy.dot(alpha[i - 1, :], P)
                alpha[i, :] = alpha[i, :] * B[i, :]
                scale[i, 0] = numpy.sum(alpha[i, :])
                alpha[i, :] = alpha[i, :] / scale[i, 0]
                
            beta[int(T[0, n]) - 1, :] = numpy.ones(shape=(1, K)) / scale [int(T[0, n]) - 1, 0]
                
            for i in range(int(T[0, n]) - 2, -1, -1):  # Starting from the second last index and counting down to 0
                beta[i, :] = numpy.dot((beta[i + 1, :] * B[i + 1, :]), P.transpose()) / scale[i, 0]
                
            gamma = (alpha * beta) + epsilon
            gamma = rDiv(gamma, rSum(gamma))
            
            gammasum = sum (gamma)
            #print(beta)
            
            for i in range(int(T[0, n])):
                # find the letter in the alphabet
                crit = (Xcurrent[i] == bins)
                m = numpy.where(crit)[0][0]
                gammaksum[m, :] = gammaksum[m, :] + gamma [i, :]
         
            for i in range(int(T[0, n]) - 1):
                alphaTrans = numpy.zeros(shape=(1, len(alpha[i, :])))
                alphaTrans[0, :] = alpha[i, :]
                #alphaTrans = alphaTrans.transpose()
                
                betaMB = numpy.zeros(shape=(len(beta[i + 1, :]), 1))
                betaMB[:, 0] = beta[i + 1, :] * B[i + 1, :]
                
                fin = numpy.dot(alphaTrans, betaMB)
                
                t = P * fin #P.multiply(fin)
                t = t / numpy.sum(t)
                sxi = sxi + t 
                
            #sxi = sparse.lil_matrix(sxi)
                
            Gammainit = Gammainit + gamma[0, :]
            Gammasum = Gammasum + gammasum
            Gammaksum = Gammaksum + gammaksum
            
            for i in range(int(T[0, n]) - 1):
                Scale[i, :] = Scale[i, :] + numpy.log(scale[i, :])

            Scale[int(T[0, n]) - 1, :] = Scale[int(T[0, n]) - 1, :] + numpy.log(scale[int(T[0, n]) - 1, :])
    
        # M Step
        
        # outputs
        E = cDiv(Gammaksum, Gammasum)

        # Transition matrix
        #sxi = sxi.toarray()
        
        #P = sparse.lil_matrix(self.rDiv(sxi, self.rSum(sxi)))
        P = rDiv(sxi, rSum(sxi))
        P = numpy.dot(P, numpy.identity(P.shape[0]))
        
        # Priors
        Pi = Gammainit[0, :] / numpy.sum(Gammainit)
        
        oldlik = lik
        lik = numpy.sum(Scale)
        LL.append(lik)
        
        print("Cycle %d log likelihood = %.3f " % (cycle + 1, lik))
        if cycle < 2:
            likbase = lik
        elif lik < (oldlik - 1e-6):
            print("vionum_binstion")
        elif ((lik - likbase) < ((1 + tol) * (oldlik - likbase))) or math.isinf(lik):
            print("END...")
            break
        
    return (E, P, Pi, LL)

if __name__ == '__main__':
    main ()