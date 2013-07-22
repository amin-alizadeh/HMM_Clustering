'''
Created on Jul 20, 2013

@author: Amin
'''
import datareader

gesture = 'l'

def main():
    print("Test")
    
    data = datareader.datareader("data", gesture)
    train = data.get_test_xyz()
    print(train)
    
if __name__ == '__main__':
    main()