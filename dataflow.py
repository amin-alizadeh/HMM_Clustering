'''
Created on Jul 22, 2013

@author: Amin
'''
import os
import csv
import numpy

class dataflow(object):
    '''
    classdocs
    '''
    path = ""
    name = ""
    D = 3
    def __init__(self, path, name):
        '''
        Constructor
        '''
        self.path = path + os.sep
        self.name = name

    def get_train_xyz(self):
        return self.get_xyz_data(self.path + 'train', self.name)

    def get_test_xyz(self):
        return self.get_xyz_data(self.path + 'test', self.name)
    
    def get_xyz_data (self, path, name):
        fx = open(path + os.sep + name + '_x.csv', 'r')
        fy = open(path + os.sep + name + '_y.csv', 'r')
        fz = open(path + os.sep + name + '_z.csv', 'r')
        
        cx = csv.reader (fx, delimiter=',', quotechar='|')
        cy = csv.reader (fy, delimiter=',', quotechar='|')
        cz = csv.reader (fz, delimiter=',', quotechar='|')
        
        x = numpy.asarray(list(cx), dtype = 'float')
        y = numpy.asarray(list(cy), dtype = 'float')
        z = numpy.asarray(list(cz), dtype = 'float')
        
        fx.close()
        fy.close()
        fz.close()
        
        i, j = x.shape
        m = numpy.empty(shape = (i, j, self.D))
        
        m[:,:, 0] = x
        m[:,:, 1] = y 
        m[:,:, 2] = z
        return m
    
    def get_test(self, path):
        tF = open(path, 'r')
        cT = csv.reader (tF, delimiter=',', quotechar='|')
        t = numpy.asarray(list(cT), dtype = 'float')
        tF.close
        return t
    
    def loadTest(self, path, files):
        testSets = {}
        for file in files:
            tst = self.get_test(path + file)
            testSets[file.partition(".csv")[0]] = tst
        return testSets

    def get_tests_attached(self, *tests):
        z = numpy.empty(shape = (tests[0].shape[0], len(tests), tests[0].shape[1]))
        i = 0
        for test in tests:
            z[:,i,:] = test
            i = i + 1
        return z

    def load_model(self):
        Ef = open(self.path + "model" + os.sep + self.name + ".csv", "r")
        ar = csv.reader (Ef, delimiter=',', quotechar='|')
        count = 0
        for row in ar:
            if count == 0:
                i = 0
                for item in row:
                    if i == 0:
                        N = int(item) # Number of states
                    else:
                        M = int(item) # Number of symbols
                        E = numpy.empty(shape = (N, M))
                        P = numpy.empty(shape = (M, M))
                        Pi = numpy.empty(M)
                        cent = numpy.empty(shape = (N, 3))
                    i = i + 1
            elif count <= N:
                i = 0
                for item in row:
                    E[count - 1, i] = float(item)
                    i = i + 1
            elif count > N and count <= N + M:
                i = 0
                for item in row:
                    P[count - N - 1, i] = float(item)
                    i = i + 1
            elif count == N + M + 1:
                Pi = numpy.asarray(list(row), dtype = 'float')
            elif count > N + M + 1 and count < N + M + N + 2:
                i = 0
                for item in row:
                    cent[count - N - M - 2, i] = float(item)
                    i = i + 1
            elif count == N + M + N + 2:
                for item in row:
                    thr = float (item)
            count = count + 1
        Ef.close()

        return (E, P, Pi, cent, thr)
        
    def store_model(self, E, P, Pi, cent, thr):
        Ef = open(self.path + "model" + os.sep + self.name + ".csv", "w")
        Ewriter = csv.writer(Ef, delimiter = ',', quotechar = '', quoting = csv.QUOTE_NONE, dialect = csv.unix_dialect)
        Ewriter.writerow(E.shape)
        Ewriter.writerows(E)
        Ewriter.writerows(P)
        Ewriter.writerow(Pi)
        Ewriter.writerows(cent)
        Ewriter.writerow([thr])
        Ef.close()

    def store_Binned(self, TrainBinned, TestBinned):
        Ef = open(self.path + "Binned" + os.sep + self.name + ".csv", "w")
        Ewriter = csv.writer(Ef, delimiter = ',', quotechar = '', quoting = csv.QUOTE_NONE, dialect = csv.unix_dialect)
        Ewriter.writerows(TrainBinned)
        Ewriter.writerows(TestBinned)
        Ef.close()
        