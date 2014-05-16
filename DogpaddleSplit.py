
import os
import sys
import csv
import numpy
import RecordingAnalyzer

def main():
    root_path = "data" + os.sep + "unitytrain"
    gesture_name = "dogpaddle"
    orient = "R"
    joint_header = "Hand_" + orient.upper()
    HL = "Hand_L"
    HR = "Hand_R"
    all_sequences = RecordingAnalyzer.get_All_Files(root_path + os.sep + gesture_name + "-" + orient.lower())
    sequences = []
    path_p1 = root_path + os.sep + gesture_name + "-" + orient.lower() + os.sep + "p1"
    path_p2 = root_path + os.sep + gesture_name + "-" + orient.lower() + os.sep + "p2"
    if not os.path.isdir(path_p1):
        os.mkdir(path_p1)
    if not os.path.isdir(path_p2):
        os.mkdir(path_p2)
        
    for file in all_sequences:
        p1, p2 = split_sequence(RecordingAnalyzer.get_xyz_data(file)[joint_header], axis = 1, Min = False)
        fl = file.split('\\')
        fn = fl[len(fl) - 1]
        store_sequence(p1, joint_header, path_p1, fn)
        store_sequence(p2, joint_header, path_p2, fn)
        

def store_sequence(data, header, path, file_name):
    Ef = open(path + os.sep + file_name, "w")
    Ewriter = csv.writer(Ef, delimiter = ',', quotechar = '', quoting = csv.QUOTE_NONE, dialect = csv.unix_dialect)
    Ewriter.writerow([header,"",""])
    Ewriter.writerows(data)
    Ef.close()

    
def split_sequence(sequence, axis = 1, Min = True):
    if Min:
        split_index = numpy.argmin(sequence[:,axis])
    else:
        split_index = numpy.argmax(sequence[:,axis])
    
    length = len(sequence)
    s1 = sequence[:split_index, :]
    s2 = sequence[split_index:length, :]
    return s1, s2 
    
              
if __name__ == '__main__':
    main()