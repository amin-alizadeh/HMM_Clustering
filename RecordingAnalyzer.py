import os
import sys
import csv
import numpy
import math
from dataflow import dataflow
from numpy.numarray.numerictypes import Int32

centroids_path = "all-centroids.csv"
root_path = "data" + os.sep + "unitytrain"
gesture_name = "crawl-l"
orient = "L"
joint_header = "Hand_" + orient
HL = "Hand_L"
HR = "Hand_R"
parent_joint_header = "Shoulder_" + orient
D = 3  # the number of dimensions to use: X, Y, Z
M = 12  # output symbols
N = 5  # states
LR = 2  # degree of play in the left-to-right HMM transition matrix 
all_gesture_names = [["circle-l-ccw", HL], ["circle-l-cw", HL], ["circle-r-ccw", HR], ["circle-r-cw", HR], \
                     ["crawl-l", HL], ["crawl-r", HR], ["fly", HL], ["fly", HR], ["frog", HL], ["frog", HR], \
                     ["wave-l", HL], ["wave-r", HR], ["dog-rdown-lup", HL], ["dog-rdown-lup", HR], \
                     ["dog-rup-ldown", HL], ["dog-rup-ldown", HR]] 
all_gesture_names = [["dog-rup-ldown", HL]]
saveModel = False
def main ():
    training_path = root_path + os.sep + gesture_name
    allFiles = get_All_Files (training_path)

    all_trains = [] #Array of dictionaries. Each dictionary contains coordinates of  
                    #the joints. The name of the joints are the keys of the dictionaries
    #print(allFiles)
#     return
    for file in allFiles:
        all_trains.append(get_xyz_data(file))
    
    training_path = root_path + os.sep + gesture_name
    allFiles = get_All_Files (training_path)

    all_trains = [] #Array of dictionaries. Each dictionary contains coordinates of  
                    #the joints. The name of the joints are the keys of the dictionaries
    #print(allFiles)
#     return
    for file in allFiles:
        all_trains.append(get_xyz_data(file))
    

    
    joints = put_joints_together (all_trains)
    
    normalized_joint = {}
    
    normalized_joint[joint_header] =  normalize_joint_with_parent (joints, joint_header, parent_joint_header)
    
    centroids = get_point_centroids (normalized_joint, N)
    print(centroids)
    Ef = open(root_path + os.sep + gesture_name + os.sep + "centroids.csv", "w")
    Ewriter = csv.writer(Ef, delimiter = ',', quotechar = '', quoting = csv.QUOTE_NONE, dialect = csv.unix_dialect)
    Ewriter.writerow([joint_header])
    Ewriter.writerows(centroids[joint_header])
    Ef.close()
    return
    for gesture_joint in all_gesture_names:
        train_model(gesture_joint[0], gesture_joint[1], saveModel)
    
def train_model(gesture_name, joint_header, save = True):
    print("Running for ", gesture_name, joint_header, "...")
    training_path = root_path + os.sep + gesture_name
    allFiles = get_All_Files (training_path)

    all_trains = [] #Array of dictionaries. Each dictionary contains coordinates of  
                    #the joints. The name of the joints are the keys of the dictionaries
    #print(allFiles)
#     return
    for file in allFiles:
        all_trains.append(get_xyz_data(file))
    
    directions = get_directed_vectors()

#     print(len(directions), directions)
#     return
    N = len(directions)
    M = int(N + (N / 3))
    train_directed = get_directed_points(all_trains, joint_header)
    
    train_binned_dirs = get_point_clusters_with_directions(train_directed, directions)
    print(len(allFiles), len(train_directed), len(train_binned_dirs))
    #return
#     for i in range(len(train_binned_dirs)):
#         print(i + 1, len(train_binned_dirs[i]), train_binned_dirs[i], train_directed[i])
#     return
    """
    joints:
    Dictionary: keys -> joints such as "Body", "Hand_R", etc.
                values -> List of all coordinates
                        elements of the list -> Numpy array of the coordinates
    """
    '''
    joints = put_joints_together (all_trains)
    
    normalized_joint = {}
    
    normalized_joint[joint_header] =  normalize_joint_with_parent (joints, joint_header, parent_joint_header)
    
    centroids = get_point_centroids (normalized_joint, N)
    train_binned = get_point_clusters(normalized_joint, centroids)
    '''
    """
    print (centroids[joint_header])
    store_centroids(centroids[joint_header], root_path, gesture_name)
    return
    
    centroids = get_xyz_data_no_header(root_path + os.sep + centroids_path)
    print(centroids)
    
    train_binned = get_point_clusters(normalized_joint, centroids)
    print(train_binned)
    return
    """
    """
    for key in train_binned.keys():
        for item in train_binned[key]:
            print(key, ": ", item.shape)

    """
    '''
    ****************************************************
    *  Training
    ****************************************************
    '''
    
    # Set priors
    pP = prior_transition_matrix(M, LR)
    
    # Train the model:
    b = [x for x in range(N)]
    cyc = 60
    '''
    training_data_binned = train_binned[joint_header]
    '''
    training_data_binned = train_binned_dirs
    
    E, P, Pi, LL = dhmm_numeric(training_data_binned, pP, b, M, cyc, .00001)
    
    sumLik = 0
    minLik = numpy.Infinity

    for i in range(len(training_data_binned)):
        lik = pr_hmm(training_data_binned[i], P, E, Pi)
        if lik < minLik:
            minLik = lik
        sumLik = sumLik + lik
    
    gestureRecThreshold = 2.0 * sumLik / len(training_data_binned)
    
    print('\n********************************************************************')
    print('Testing %d sequences for a log likelihood greater than %.4f' % (1, gestureRecThreshold))
    print('********************************************************************\n')
    recs = 0
    for one_training in training_data_binned:
        score = pr_hmm(one_training, P, E, Pi)
        if score > gestureRecThreshold:
            #print("Log Likelihood: %.3f > %.3f (threshold) -- FOUND %s Gesture" % (score, gestureRecThreshold, gesture_name))
            recs = recs + 1
#         else:
#             print("Log Likelihood: %.3f < %.3f (threshold) -- NO %s Gesture" % (score, gestureRecThreshold, gesture_name))
    
    print('Recognition success rate: %.2f percent\n' % (100 * recs / len(training_data_binned)))

    
    """ Uncomment the following line to store the model"""
    if save:
        store_model(E, P, Pi, gestureRecThreshold, root_path, gesture_name + joint_header)
    #dtf = dataflow(root_path, gesture_name)
    #dtf.store_model(E, P, Pi, centroids[joint_header], gestureRecThreshold)

def store_model(E, P, Pi, thr, path, name):
    Ef = open(path + os.sep + "model" + os.sep + name + ".csv", "w")
    Ewriter = csv.writer(Ef, delimiter = ',', quotechar = '', quoting = csv.QUOTE_NONE, dialect = csv.unix_dialect)
    Ewriter.writerow(E.shape)
    Ewriter.writerows(E)
    Ewriter.writerows(P)
    Ewriter.writerow(Pi)
    Ewriter.writerow([thr])
    print("Saved successfully.")
    Ef.close()
    
def get_directed_points(points, header):
    all_directed = []
    it = 0
    for point_set in points:
        it = it + 1
        p = point_set[header]
        vecs = []
        last_p = p[0, :]
        #print("iteration", it)
        for i in range(len(p) - 1):
            current_p = p[i + 1,:] 
            vec = (current_p - last_p)
            mag_p = numpy.linalg.norm(vec)
            #print(mag_p)
            if (mag_p > 0.05):
                vecs.append(vec)
                last_p = current_p
        print(len(vecs))
        if (len(vecs) > 0):
            d = numpy.asarray(list(vecs), dtype = 'float')
            all_directed.append(d)
    return all_directed
    
    
def get_point_clusters_with_directions (points, dirs):
    all_clustered = []
    for point_set in points:
        length = len(point_set)
        clusters = []
        for i in range(length):
            if numpy.sqrt(point_set[i,:].dot(point_set[i,:])) > 0.07:
                clusters.append(assign_point_to_cluster(point_set[i,:], dirs))
        xClustered = numpy.asarray(list(clusters), dtype = 'float')
        all_clustered.append(xClustered)
    return all_clustered

def assign_point_to_cluster(vec, dirs):
    if vec[0] < 0:
        if vec[1] < 0:
            return get_cluster_direction_dot(vec, dirs[24:]) + 24
        else:
            return get_cluster_direction_dot(vec, dirs[18:24]) + 18
    else:
        return get_cluster_direction_dot(vec, dirs[:24])
    
def get_cluster_direction_dot(vec, p):
    vec = normalize_vector(vec)
#     if numpy.linalg.norm(vec) == 0:
#         return 13
     
    max_dist = -2
    max_dist_index = -1
    for i in range(len(p)):
        d = numpy.dot(vec, p[i])
        if d > max_dist:
            max_dist = d
            max_dist_index = i
    return max_dist_index
    
     
def get_directed_vectors():
    p = []
    for i in range(1,-2,-1):
        for j in range(1,-2,-1):
            for k in range(1,-2,-1):
                if i != 0 or j != 0 or k != 0:
                    p.append(normalize_vector(numpy.array([i, j, k])))
    return p

def normalize_vector(vec):
    nrm = numpy.linalg.norm(vec)
    if nrm == 0:
        return vec
    else:
        return vec / nrm
        
def store_centroids (centroids, path, name):
    Ef = open(path + os.sep + "centroids" + os.sep + name + ".csv", "w")
    Ewriter = csv.writer(Ef, delimiter = ',', quotechar = '', quoting = csv.QUOTE_NONE, dialect = csv.unix_dialect)
    Ewriter.writerows(centroids)
    Ef.close()
    print("Centroids stored to file %s successfully..." %(name))
    
def normalize_joint_with_parent (joints, header, parent):
    normalized_header = []
    for i in range(len(joints[header])):
        normalized = joints[header][i] - joints[parent][i]
        normalized_header.append(normalized)
    
    return normalized_header
   
def pr_hmm(clusters, P, E, pi):
    """
    INPUTS:
    O=Given observation sequence labeled in numerics
    A(N,N)=transition probability matrix
    B(N,M)=Emission matrix
    pi=initial probability matrix
    Output
    P=probability of given sequence in the given model
    
    Copyright (c) 2009, kannu mehta
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without 
    modification, are permitted provided that the following conditions are 
    met:
    * Redistributions of source code must retain the above copyright 
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright 
    notice, this list of conditions and the following disclaimer in 
    the documentation and/or other materials provided with the distribution
           
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
    POSSIBILITY OF SUCH DAMAGE.
    """
    # Clusters, P, E , Pi
    # clusters       , P, E , pi
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
        p = math.log(p, 2)
    else:
        p = float('-inf')
        
    return p

def get_point_centroids (data, N):
    headers = list(data.keys())
    
    number_of_rows = 0
    
    for item in data[headers[0]]:
        number_of_rows = number_of_rows + len(item)
    
    centroids = {}
    
    for header in headers:
        merged = numpy.empty(shape = (number_of_rows, 3))
        this_row = 0
        for item in data[header]:
            length = len(item)
            merged[this_row: this_row + length, :] = item
            this_row = this_row + length
        centroids[header], points, idx = kmeans(merged, N)
    
    return centroids
    


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
    
    for header in headers:
        each_joint = []
        for t in all_joints:
            each_joint.append(t[header])
        joints[header] = each_joint
    
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

def get_xyz_data_no_header(path):
    f = open(path)
    c = csv.reader (f, delimiter=',', quotechar='|')
    d = numpy.asarray(list(c), dtype = 'float')
    f.close()
    return d

def get_point_clusters(all_data, all_centroids):
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
                Z[n] = numpy.sum(X[n, :])
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
    
    cycles = 0
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
        
        #print("Cycle %d log likelihood = %.3f " % (cycle + 1, lik))
        if cycle < 2:
            likbase = lik
        elif lik < (oldlik - 1e-6):
            print("vionum_binstion")
        elif ((lik - likbase) < ((1 + tol) * (oldlik - likbase))) or math.isinf(lik):
            print("END...")
            break
        cycles = cycle + 1
        
    print("Iterated for %d cycles" %(cycles))    
    return (E, P, Pi, LL)


  
if __name__ == '__main__':
    main ()