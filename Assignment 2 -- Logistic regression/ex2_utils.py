import os 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import MaxNLocator

__author__ = "VietTra"
__email__ = "viettra95@gmail.com"
__date__ = "04th_May_2018"

def load(file_path):
    """ This function reads a .txt file and return a numpy array.
    
    Arguments:
    - file_path: str -- the location path of the txt file
    
    Returns:
    - data: numpy array -- the loaded data
    
    Raises:
    - None   
    """
    if os.path.exists(file_path) == False:
        return None
    with open(file_path, 'rb') as f:
        file_lines = f.readlines()
    
    file_lines = [file_line.decode("utf-8")[:-1] for file_line in file_lines]

    data = np.zeros((len(file_lines), file_lines[0].count(',')+1))
    for i, file_line in enumerate(file_lines):
        nums = file_line.split(',')
        data[i,:] = [float(num) for num in nums]
    return data    


def plot_data(X, y):
    """
    This function plots the data points X and y into a new figure, 
    gives the figure axes labels of population and profit, plots the data 
    points with + for the positive examples and o for the negative examples.
    X is assumed to be a Mx2 matrix.

    Arguments:
    - X,y: np.ndarray -- input data to plot.
    
    Raises:
    - None
    
    Return:
    - fig

    """
       
    fig = plt.figure()
    
    neg = np.where(y==0)[0].reshape((-1,1))
    pos = np.where(y==1)[0].reshape((-1,1))
    
    plt.plot(X[pos,0], X[pos,1], 'k+', lw=2, ms=7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko',mfc='y', ms=7)

    return fig

def sigmoid(z):
    """
    Function sigmoid(z) computes the sigmoid of z.
    
    Arguments:
    - z: input value to compute, can be a array, vector or scalar.
    
    Raises:
    - None
    
    Returns:
    - g: computed sigmoid value.
    
    """

    g = 1/(1 + np.exp(-z))
    
    return g


class CostFunc(object):
    """
    This class provides two function `cost` and `gradient` which compute cost and gradient
    for Logistic Regression
    """
    def __init__(self, X=None, y=None):
        assert X is not None, \
            "The argument `X` must be specified!"
        assert y is not None, \
            "The argument `y` must be specified!"        
        self.X = X
        self.y = y
        self.m = len(y)
        
    def cost(self, theta):
        """
        This function computes cost for logistic regression
        
        Arguments:
        - self.(X, y): np.ndarray -- input data
        - theta: ndarray, optional 

        Returns
        - J: float -- Computed cost.

        Raises:
        - None
        """
        X = self.X
        y = self.y
        m = self.m        
        
        theta = theta.reshape((-1,1))
        
        h = sigmoid(np.dot(X, theta))
        J = np.sum(np.dot(-y.T, np.log(h)) - np.dot((1-y).T, np.log(1-h)), axis=0)/m
        return float(J)
    
    def gradient(self, theta):
        """
        This function computes the gradient of the cost for Regularized logistic regression
        
        Arguments:
        - self.(X, y): np.ndarray -- input data
        - theta: ndarray, optional 

        Returns
        - grad: ndarray -- Computed gradient.

        Raises:
        - None
        """
        X = self.X
        y = self.y
        m = self.m
        
        if len(theta.shape) == 1:
            theta = theta.reshape((-1,1))
        
        h = sigmoid(np.dot(X, theta))
        #a = np.repeat((h-y), X.shape[-1], axis=1)    
        #print("a shape: {}".format(a.shape))
        #print("X shape -1: {}".format(X.shape[-1]))        
        grad = np.dot(X.T, h-y)/m
        return grad
      

def map_feature(X1, X2):
    """
    This function maps the two input features to quadratic features 
    used in the regularization exercise.
    
    Arguments:
    - X1, X2: ndarray -- inpute features
    
    Raises:
    - None.
    
    Returns: 
    - out: a new feature array with more features, comprising of 
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    """
    
    #Inputs X1, X2 must be the same size
    
    degree = 6
    if X1.shape != ():
        out = np.ones((X1.shape[0],1))
        for i in range(1,degree+1):
            for j in range(i+1):
                out = np.append(out, (X1**(i-j))*(X2**j), axis=1)
    else:    
        out = [1]
        for i in range(1,degree+1):
            for j in range(i+1):
                out.append( (X1**(i-j)) * ((X2**j)) )
            
    return np.array(out)


def plot_decision_boundary(theta, X, y):
    """
    - This function plots the data points X and y into a new figure with
    the decision boundary defined by theta.
    - X is assumed to be a either 
        1) Mx3 matrix, where the first column is an all-ones column for the intercept.
        2) MxN, N>3 matrix, where the first column is all-ones.
    
    Arguments:
    - theta, X, y: ndarray -- input data.
    
    Raises: 
    - None.
    
    Returns:
    - None.
    """
    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([[np.min(X[:,1]-2)], [np.max(X[:,1]+2)]])
        
        # Calculate the decision boundary line
        plot_y = (-1/theta[-1])*(theta[1]*plot_x + theta[0])
        
        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)
        
        # Legend, specific for the exercise
        # legend('Admitted', 'Not admitted', 'Decision Boundary')
        plt.axis([30, 100, 30, 100])
        plt.show()
        
    else:
        #  Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))

        # Evaluate z = theta*x over the grid

        for i in range(u.shape[0]):
            for j in range(v.shape[0]):
                z[i, j] = np.dot(map_feature(u[i], v[j]), theta)
        # important to transpose z before calling contour
        z = z.T
        
        plt.contour(u, v, z, levels=[0]).collections[0].set_label('')       
    return None

def predict(theta, X, y):
    """
    This funciton predictc whether the label is 0 or 1 using learned logistic 
    regression parameters theta, uses threshold at 0.5
    
    Arguments:
    - X, y: ndarray-- input data
    - theta: ndarray -- learned logistic regression parameters theta
    
    Raises:
    - None
    
    Returns:
    - p: ndarray -- prediction results vector of 0's and 1's
    - acc: float -- train accurary
    """
    
    m = len(X)
    p = np.zeros((m,1))
    acc = 0
    
    s = sigmoid(np.dot(X,theta))
    for i in range(m):
        if s[i] >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
        if p[i] == y[i]:
            acc+=1
    return p, (acc/m)*100



class CostFuncReg(object):
    """
    This class provides two function `cost` and `gradient` which compute cost and gradient
    for Regularized Logistic Regression
    """
    def __init__(self, X=None, y=None, learningRate=None):
        assert X is not None, \
            "The argument `X` must be specified!"
        assert y is not None, \
            "The argument `y` must be specified!"
        assert learningRate is not None, \
            "The argument `Lambda` must be specified!"  
        self.X = X
        self.y = y
        self.learningRate = learningRate
        self.m = len(y)
        
    def cost(self, theta):
        """
        This function computes cost for Regularized logistic regression
        
        Arguments:
        - self.(X, y): np.ndarray -- input data
        - theta: ndarray, optional 

        Returns
        - J: float -- Computed cost.

        Raises:
        - None
        """
        X = self.X
        y = self.y
        m = self.m        
        learningRate = self.learningRate

        theta = theta.reshape((-1,1))
        
        h = sigmoid(np.dot(X, theta))

        first = (1/m)*np.sum(np.dot(-y.T, np.log(h)) - np.dot((1-y).T, np.log(1-h)), axis=0)
        second = (learningRate/(2*m))*np.sum(theta[1:]**2)
        J = first + second

        return float(J)
    
    def gradient(self, theta):
        """
        This function computes the gradient of the cost for Regularized logistic regression
        
        Arguments:
        - self.(X, y): np.ndarray -- input data
        - theta: ndarray, optional 

        Returns
        - grad: ndarray -- Computed gradient.

        Raises:
        - None
        """
        X = self.X
        y = self.y
        m = self.m
        learningRate = self.learningRate

  
        theta = theta.reshape((-1,1))
        grad = np.zeros(theta.shape)
        h = sigmoid(np.dot(X, theta)) 
        
        grad = np.dot((h-y).T, X)/m
        grad = grad.T
        grad[1:] += (learningRate/m)*theta[1:]


        return grad
    
    