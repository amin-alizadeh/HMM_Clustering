'''
Created on Mar 31, 2014

@author: AminAlizadeh
'''

import numpy
import math
import sys

class hmm_numeric(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
        
    def dhmm_numeric(self, data, pP, bins, K=12, cyc=100, tol=0.0001):
        
        print(data.shape)
        return
        num_bins = len(bins)
        epsilon = sys.float_info.epsilon
        # number of sequences
        N = len(data[:, 0, 0])
        # calculating the length of each sequence
        T = numpy.ones(shape=(1, N))
        
        for n in range(N):
            T[0, n] = len(data[0, :, 0])
        
        TMAX = T.max()
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
        E = self.cDiv(E, self.cSum(E))

        B = numpy.zeros(shape=(TMAX, K))
        
        Pi = numpy.random.rand(K)
        
        #REMOVE THE FOLLOWING
        #Pi = numpy.array([ 0.94503978,  0.97160498,  0.54553485,  0.34016857,  0.48067553,  0.94473749,  0.57018499,  0.38132459,  0.40252906,  0.61204666,  0.21091647,  0.0745824 ])
        
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
                
                # Inital values of B = Prob(output|s_i), given data data
                Xcurrent = data[n, :, :]

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
            
            #print("Cycle %d log likelihood = %.3f " % (cycle + 1, lik))
            if cycle < 2:
                likbase = lik
            elif lik < (oldlik - 1e-6):
                print("vionum_binstion")
            elif ((lik - likbase) < ((1 + tol) * (oldlik - likbase))) or math.isinf(lik):
                print("END...")
                break
            
        return (E, P, Pi, LL)
    

        