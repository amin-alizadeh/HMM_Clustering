import os
from array import array

def main():
    path = "data\\observations\\model\\"
    file = "breaststroke_elbow_l_down"
    extension = ".csv"
    fT = open (path + os.sep + file + extension, "r")
    content = fT.read()
    print(type(content))
    binaryContent = str.encode(content)
    print (type(binaryContent))
    fB = open (path + os.sep + "binary" + os.sep + file, "wb+")
    fB.write(binaryContent)
    
    print(fT.read())
    fB.close()
    fT.close()

if __name__ == '__main__':
    main ()