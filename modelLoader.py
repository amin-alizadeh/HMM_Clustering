
import os
import csv
import numpy

class Loader(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    
    def getAllModels(self, path):
        allFileNames = self.getAllFileNames(path)
        models = {}
        for name in allFileNames:
            model = {}
            model["E"], model["P"], model["Pi"], model["centroids"], model["threshold"] = \
            self.load_model(path + name)
            modelName = name.partition(".csv")[0]
            models[modelName] = model
        
        return models
        
    def getAllFileNames(self, path):
        files = []
        for file in os.listdir(path):
            files.append(file)
        return files
    
    def load_model(self, path):
        Ef = open(path, "r")
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
 

def main():
    path = "data\\observations\\model\\"
    modelLoader = Loader()
    models = modelLoader.getAllModels(path)
    print(models.keys())
    
if __name__ == '__main__':
    main ()

