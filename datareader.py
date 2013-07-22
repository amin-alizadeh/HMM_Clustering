'''
Created on Jul 22, 2013

@author: Amin
'''
import os
import csv
import numpy

class datareader(object):
    '''
    classdocs
    '''
    path = ""
    name = ""

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
        
        m = []
        return m
    