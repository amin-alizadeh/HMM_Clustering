'''
Created on Jul 24, 2013

@author: Amin
'''
import numpy
import math
import sys

class classifier(object):
    '''
    classdocs
    '''

    
    def __init__(self):
        '''
        Constructor
        '''
 

        
        
    def get_point_centroids(self, data, K, D):
        mean = numpy.zeros(shape=(data.shape[0], D))  # 60 x 3
        
        for n in range(data.shape[0]):  # 60 number of rows
            for i in range(data.shape[1]):  # 10 number of columns
                for j in range(D):  # 3
                    mean[n, j] = mean[n, j] + data[n,i,j]
#                     print(data[n,i,j])
            mean[n,:] = mean[n,:] / data.shape[1] # /10

        centroids, points, idx = self.kmeans(mean, K)
        self.centroids = centroids
        return centroids
    
    
    def kmeans(self, data, nbCluster):
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

                    centroid[c, :] = (r * data_diff) + data_min
                
                # stoppingCriterion
    #             print("Centroids: ", centroid, "\nOld:", oldPositions)
            pos_diff = sum(sum((centroid - oldPositions) ** 2))
    #         print(pos_diff)
        return (centroid, pointsInCluster, assignment)
        
    def get_point_clusters(self, data, centroids, D):
        XClustered = numpy.zeros(shape=(data.shape[1], data.shape[0], 1)) # 10 x 60
        if len(data.shape) < 3:
            tmp = numpy.empty(shape = (data.shape[0], 1, data.shape[1]))
            tmp[:,0,:] = data[:,:]
            data = tmp
        
        K = len(centroids[:, 0])

        for n in range(data.shape[0]):
            for i in range(data.shape[1]):
                temp = numpy.zeros(shape=(K))
                for j in range(K):
                    if D == 3:
                        temp[j] = math.sqrt((centroids[j, 0] - data[n, i, 0]) ** 2 + (centroids[j, 1] - data[n, i, 1]) ** 2 + (centroids[j, 2] - data[n, i, 2]) ** 2)
                
                idx, I = min((val, idx) for (idx, val) in enumerate(temp))
                XClustered[i, n, 0] = I
        
        return XClustered

    def prior_transition_matrix(self, K, LR):
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

    def dhmm_numeric(self, X, pP, bins, K=2, cyc=100, tol=0.0001):
        """ simple Hidden Markov Model - variable lengths and discrete symbols
        
         X - N x p array of Xs, p sequences
         bins - possible bins of Xs
         K - number of states (default 2)
         cyc - maximum number of cycles of Baum-Welch (default 100)
         tol - termination tolerance (prop change in likelihood) (default 0.0001)
        
         E - observation emission probabilities
         P - state transition probabilities
         Pi - initial state prior probabilities
         LL - log likelihood curve
        
         Iterates until a proportional change < tol in the log likelihood 
         or cyc steps of Baum-Welch
        """
        D = len(X[0, 0, :])
        

        num_bins = len(bins)
        epsilon = sys.float_info.epsilon
        # number of sequences
        N = len(X[:, 0, 0])
        # calculating the length of each sequence
        T = numpy.ones(shape=(1, N))
        
        for n in range(N):
            T[0, n] = len(X[0, :, 0])
        
        TMAX = T.max()
        
        print('********************************************************************');
        print('Training %d sequences of maximum length %d from an alphabet of size %d' % (N, TMAX, num_bins));
        print('HMM with %d hidden states' % K);
        print('********************************************************************');
        
        rd = numpy.random.rand(num_bins, K)
        
        E = 0.1 * rd
        E = E + numpy.ones(shape=(num_bins, K))
        E = E / num_bins
        E = self.cDiv(E, self.cSum(E))

        B = numpy.zeros(shape=(TMAX, K))
        
        Pi = numpy.random.rand(K)
        
        Pi = Pi / numpy.sum(Pi)

        # transition matrix
        P = pP
        P = self.rDiv(P, self.rSum(P))
        
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
                gamma = self.rDiv(gamma, self.rSum(gamma))
                
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
            E = self.cDiv(Gammaksum, Gammasum)

            # Transition matrix
            #sxi = sxi.toarray()
            
            #P = sparse.lil_matrix(self.rDiv(sxi, self.rSum(sxi)))
            P = self.rDiv(sxi, self.rSum(sxi))
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
    
        
    def pr_hmm(self, clusters, P, E, pi):
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
        # Clusters, P, E.transpose(), Pi
        # clusters       , P, E            , pi

        n = P.shape[1]
        T = len(clusters)
        m = numpy.zeros(shape=(T, n))
        # it uses forward algorithm to compute the probability
        for i in range(n):  # initilization
            m[0, i] = E[i, int(clusters[0])] * pi[i]
            
        for t in range(T - 1):  # recursion
            for j in range(n):
                z = 0
                for i in range(n):
                    z = z + P[i, j] * m[t, i]
                m[t + 1, j] = z * E[j, int(clusters[t + 1])]
            
        p = 0
        
        for i in range(n):  # termination
            p = p + m[T - 1, i]
        
        if p != 0:
            p = math.log(p)
        else:
            p = float('-inf')
            
        return p
