import os
import sys
import csv
import numpy
import math
from hmm_numeric import hmm_numeric
root_path = "data" + os.sep + "unitytrain"
gesture_name = "circle-r-cw"
D = 3  # the number of dimensions to use: X, Y, Z
M = 12  # output symbols
N = 8  # states
LR = 2  # degree of play in the left-to-right HMM transition matrix 

def main ():
    training_path = root_path + os.sep + gesture_name
    allFiles = get_All_Files (training_path)
    all_trains = [] #Array of dictionaries. Each dictionary contains coordinates of  
                    #the joints. The name of the joints are the keys of the dictionaries
    for file in allFiles:
        all_trains.append(get_xyz_data(file))
    
    joints = put_joints_together (all_trains)
    centroids = {}
    train_binned = {}
    joint_header = "Hand_R"
    centroids[joint_header], points, idx = kmeans(joints[joint_header], N)
    train_binned[joint_header] = get_point_clusters(joints[joint_header], centroids[joint_header])
    
    print(train_binned[joint_header].shape)

    '''
    ****************************************************
    *  Training
    ****************************************************
    '''
    
    # Set priors
    pP = prior_transition_matrix(M, LR)
    
     # Train the model:
    b = [x for x in range(N)]
    cyc = 50
    print(train_binned[joint_header].shape)
    dhmm_numeric(train_binned[joint_header], pP, b, M, cyc, .00001)
    #E, P, Pi, LL = hmm_numeric.dhmm_numeric(train_binned[joint_header], pP, b, M, cyc, .00001)

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

def put_joints_together (all_joints):
    if len(all_joints) < 1:
        return {}
    
    headers = list(all_joints[0].keys())
    joints = {}
    number_of_rows = 0
    for entry in all_joints:
        number_of_rows += len(entry[headers[0]])
    
    
    for header in headers:
        j = numpy.zeros(shape = (number_of_rows, 3))
        rows_filled = 0
        for entry in all_joints:
            this_rows = len(entry[header])
            j[rows_filled:rows_filled + this_rows, : ] = entry[header]
            rows_filled = rows_filled + this_rows
        joints[header] = j
        
    return joints 

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

def get_point_clusters(data, centroids):
    length = len(data)
    XClustered = numpy.empty(length)
    D = 3
    number_of_clusters = len(centroids)

    for i in range(length):
        temp = numpy.zeros(shape=(number_of_clusters))
        for j in range(number_of_clusters):
            temp[j] = math.sqrt((centroids[j, 0] - data[i, 0]) ** 2 + \
                                (centroids[j, 1] - data[i, 1]) ** 2 + \
                                (centroids[j, 2] - data[i, 2]) ** 2)
        
        idx, I = min((val, idx) for (idx, val) in enumerate(temp))
        XClustered[i] = I

    return XClustered

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
    data_min = [min(data[:, 0]), min(data[:, 1]), min(data[:, 2])]
    data_max = [max(data[:, 0]), max(data[:, 1]), max(data[:, 2])]
    data_diff = numpy.subtract(data_max, data_min)

    # every row is a centroid
    centroid = numpy.random.rand(nbCluster, data_dim)
    
    """
    # REMOVE THE FOLLOWING
    cent = numpy.matrix('0.47476265  0.00958586  0.77111729; 0.64059689  0.24520689  0.15797167;  0.68462691  0.48319283  0.0998451;  0.72207494  0.67570182  0.13977921;  0.72224763  0.62701183  0.25630908;  0.71363896  0.10502063  0.57669729;  0.32202668  0.64455021  0.6286159; 0.5545931   0.63787338  0.78358486')
    centroid = numpy.empty(shape = (nbCluster, data_dim))
    for i in range(nbCluster):
        for j in range(data_dim):
            centroid[i,j] = cent[i,j]
#         print(centroid[0,:])
#         print(data_min)
#         print(centroid[0, :] + data_min)
#         print(centroid[0, 0] + data_min[0])
    # <----- REMOVE UNTIL HERE
    """
    for i in range(len(centroid[:, 1])):
        centroid [i, :] = centroid[i, :] * data_diff
        centroid [i, :] = centroid[i, :] + data_min

    # end init centroids
    
    # no stopping at start
    pos_diff = 1.0
    
    # main loop
    while pos_diff > 0.0:
        
        # E-step
        assignment = numpy.empty(len(data[:, 0]))
        
        # assign each datapoint to the closest centroid
        for d in range(len(data[:, 0])):
            min_diff = numpy.subtract(data[d, :], centroid[0, :])
            min_diff = numpy.dot(min_diff, min_diff.transpose())
            curAssignment = 0
            
            for c in range(1, nbCluster):
                diff2c = numpy.subtract(data[d, :], centroid[c, :])
                diff2c = numpy.dot(diff2c, diff2c.transpose())
                if min_diff >= diff2c:
                    curAssignment = c
                    min_diff = diff2c
            
            # assign the d-th dataPoint
            assignment[d] = curAssignment
            # for the stoppingCriterion
        oldPositions = centroid
            
        # M-Step
        # recalculate the positions of the centroids
        
        centroid = numpy.zeros(shape=(nbCluster, data_dim))
        pointsInCluster = numpy.zeros(shape=(nbCluster, 1))
        
        for d in range(len(assignment)):
            centroid[assignment[d], :] = centroid[assignment[d], :] + data[d, :]
            pointsInCluster[assignment[d], 0] = pointsInCluster[assignment[d], 0] + 1
        
        # pointsInCluster
        for c in range(nbCluster):
            
            if pointsInCluster[c, 0] != 0:
                centroid[c, :] = centroid[c, :] / pointsInCluster[c, 0]
            else:
                # set cluster randomly to new position
                r = numpy.random.rand(1, data_dim)
                """
                # REMOVE FROM HERE ->
                r[0,0] = 0.82669491
                r[0,1] = 0.48544819
                r[0,2] = 0.76186241
                # <- UNTIL HERE
                """
                centroid[c, :] = (r * data_diff) + data_min
            
            # stoppingCriterion
#             print("Centroids: ", centroid, "\nOld:", oldPositions)
        pos_diff = sum(sum((centroid - oldPositions) ** 2))
    #         print(pos_diff)
        return (centroid, pointsInCluster, assignment)

def rDiv (self, X, Y):
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

def rSum(self, X):
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
 
def cSum (self, X):
    '''
       Column sum
    '''
    Z = numpy.zeros(shape=(1, len(X[0, :])))
    if len(X[:, 0]) > 1:
        Z[0, :] = numpy.sum(X, axis=0)
    else:
        Z[0, :] = X
        
    return (Z)


def cDiv(self, X, Y):
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

    

def dhmm_numeric(X, pP, bins, K=12, cyc=100, tol=0.0001):
    
    print(X.shape)
    return
    num_bins = len(bins)
    epsilon = sys.float_info.epsilon
    # number of sequences
    N = len(X[:, 0, 0])
    # calculating the length of each sequence
    T = numpy.ones(shape=(1, N))
    
    for n in range(N):
        T[0, n] = len(X[0, :, 0])
    
    TMAX = len(X)
    """
    print('********************************************************************');
    print('Training %d sequences of maximum length %d from an alphabet of size %d' % (N, TMAX, num_bins));
    print('HMM with %d hidden states' % K);
    print('********************************************************************');
    """
    rd = numpy.random.rand(num_bins, K)
    
    """
    #REMOVE THE FOLLOWING
    r =  numpy.matrix('0.13840742  0.37774919  0.73971265  0.10260062  0.52799844  0.84634679  0.95247194  0.55089368  0.7634405   0.95265488  0.65547449  0.14542012; 0.79203228  0.23923569  0.54542612  0.20833453  0.99105794  0.32188346   0.59534973  0.89750548  0.66170189  0.92220266  0.43715002  0.14287228; 0.8594316   0.52905046  0.94455854  0.2370784   0.67138739  0.69332352   0.63358944  0.43099664  0.54580393  0.36769975  0.57135703  0.96968013; 0.17165512  0.47319164  0.50326109  0.57501913  0.15378144  0.21838713   0.29350008  0.85767455  0.41370775  0.69666658  0.10536083  0.38186331; 0.15476522  0.51562812  0.48478556  0.34682423  0.17469829  0.36680453   0.78796097  0.29453568  0.68088392  0.20358457  0.35758929  0.78457648; 0.03785457  0.18887611  0.869368    0.57801324  0.85593098  0.46747172   0.01598077  0.0152702   0.47954854  0.71670512  0.54985305  0.43920347; 0.62782384  0.16444154  0.5743267   0.78273275  0.40120534  0.9502203   0.25755689  0.89827889  0.571113    0.4517351   0.19263601  0.04172249; 0.48224823  0.28270609  0.43049263  0.14477011  0.77602924  0.19677031   0.62952399  0.69589213  0.47224108  0.5373157   0.56123885  0.22894984')
    rd = numpy.empty(shape = (num_bins, K))
    for i in range(num_bins):
        for j in range(K):
            rd[i,j] = r[i,j]
    # REMOVE UTIL HERE
    """
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
            
            # Inital values of B = Prob(output|s_i), given data X
            Xcurrent = X[n, :, :]

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
        
        #print("Cycle %d log likelihood = %.3f " % (cycle + 1, lik))
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