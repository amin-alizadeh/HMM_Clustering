import csv
import numpy
import os
from pylab import *

def main ():
    path = 'data\\kids\\Bahar'
    name = '5'
    gesture = 'elbow'
    axis = 0
    
    tr = get_data(path, name)
    
    hand_L = tr[:,0:3]
    hand_R = tr[:,3:6]
    elbow_L = tr[170:,6:9]
    elbow_R = tr[170:,9:12]
    
    #plot_left_right(hand_L, hand_R, axis, gesture)
    plot_left_right(elbow_L, elbow_R, axis, gesture)
    #slices = slice_on_min_max(hand_L, axis = 2)

def get_all (path):
    d = {}
    files = os.listdir(path)
    for file in files:
        if os.path.isfile(path + os.sep + file):
            fName = file[0:len(file) - 4]
            d[fName] = get_csv_format(path + os.sep + file)
    
    return d

def get_csv_format(path):
    f = open(path, "r")
    c = csv.reader (f, delimiter=',', quotechar='|')
    x = numpy.asarray(list(c), dtype = 'float')
    return x

def plot_left_right(left, right, axis, gesture):
    x = range(len(left))
    plot(x,left[:,axis], color = 'blue', label = 'Left ' + gesture)
    plot(x,right[:,axis], color = 'red', label = 'Right ' + gesture)
    legend(loc = 'upper right')
    xlim(0,400)
    show()

def slice_on_min_max(data, axis = 2):
    slices = []
    
    min_data = max_data = data[0, axis]
    slice_start = 0
    slice_end = 0
    up = True
    N = len(data)
    
    for i in range(1, N):
        if data[i, axis] < data[i - 1, axis]:
            up = False
        else:
            up = True
        
    return slices

def get_data(path, name):
    f = open(path + os.sep + name + ".csv", "r")
    c = csv.reader (f, delimiter=',', quotechar='|')
    x = numpy.asarray(list(c), dtype = 'float')
    
    return x
    
if __name__ == '__main__':
    main()