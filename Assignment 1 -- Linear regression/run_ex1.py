import numpy as np
import matplotlib.pyplot as plt

class Dataset(object):
    def __init__(self):
        self.data1 = 'ex1/ex1data1.txt'
        self.data2 = 'ex1/ex1data2.txt'
        self.get_data1()
        
    def get_data1(self):
        with open(self.data1, 'r') as f:
            content = f.read().splitlines()
        X = []
        y = []
        for line in content:
            line = line.split(',')
            X.append(float(line[0]))
            y.append(float(line[1]))
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        #return X, y
    
    def plot_data(self):
        print("plotting data...")
        plt.plot(self.X, self.y, 'rx')
        plt.xlabel('Populations of City in 10,000s')
        plt.ylabel('Profit in $10,000s')
        plt.show()
        
        
        
def simple_np():
    A = np.eye(5)
    print('A = {}'.format(A))

    
if __name__ == '__main__':
    simple_np()
    input('press Enter to continue')
    #print("something's happening!")
    Dataset().plot_data()
    

