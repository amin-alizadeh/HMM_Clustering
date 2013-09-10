import csv
import numpy
import os
from pylab import *

def main ():
    path = 'data\\kids\\Bahar'
    name = '5'
    gesture = 'hand'
    axis1 = 0
    axis2 = 2
    tr = get_data(path, name)
    
    hand_L = tr[:,0:3]
    hand_R = tr[:,3:6]
    elbow_L = tr[:,6:9]
    elbow_R = tr[:,9:12]
    
    path = 'data\\observations\\training\\breaststroke\\hand_l\\'
    d1 = get_data(path + 'down', 'd1')
    u1 = get_data(path + 'up', 'u1')
    plot_left_right(hand_L, hand_R, axis1, axis2, gesture)
    #plot_left_right(elbow_L, elbow_R, axis, gesture)
    #slices = slice_on_min_max(hand_L, axis = 2)


def plot_left_right(left, right, axis1, axis2, gesture):
    x = range(len(left))
    r1 = 30
    r2 = 60
    plot(left[r1:r2,axis1],left[r1:r2,axis2], color = 'blue', label = 'Left ' + gesture)
    #plot(x,right[:,axis1], color = 'red', label = 'Right ' + gesture)
    #legend(loc = 'upper right')
    #xlim(0,400)
    show()

def get_data(path, name):
    f = open(path + os.sep + name + ".csv", "r")
    c = csv.reader (f, delimiter=',', quotechar='|')
    x = numpy.asarray(list(c), dtype = 'float')
    
    return x
    
if __name__ == '__main__':
    main()