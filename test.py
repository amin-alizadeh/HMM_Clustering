
def main():
    x = []
    x.append([1,2,3])
    x.append([3,2,5])
    x.append([5,0,8])
    print(x)
    x.pop(0)
    print(x)
    x.append([3,7,1])
    print(x)
    print(len(x))
    x.pop(0)
    print(x)
    x.append([9,2,1])
    print(x)
    
if __name__ == '__main__':
    main()