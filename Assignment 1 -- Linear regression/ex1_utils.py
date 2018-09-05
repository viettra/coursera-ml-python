import os 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import MaxNLocator

__author__ = "VietTra"
__email__ = "viettra95@gmail.com"
__date__ = "26th_Apr_2018"

def load(file_path):
    """ This function reads a .txt file and return a numpy array.
    
    Arguments:
    - file_path: str -- the location path of the txt file.
    
    Returns:
    - data: numpy.ndarray -- the loaded data.
    
    Raises:
    - AssertionError: if `file_path` does not exist.   
    """
    assert os.path.exists(file_path) is True, "File path does not exist!"
    
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
    gives the figure axes labels of population and profit.
    
    Arguments:
    - X,y: numpy.ndarray -- input data to plot.
    
    Returns:
    - None.
    
    Raises:
    - None.    
    """
    fig = plt.figure()
    plt.plot (X, y, 'rx')
    plt.ylabel ('Profit in $10,000s')
    plt.xlabel ('Population of City in 10,000s')
    return fig

def compute_cost(X, y, theta):
    """
    Function to compute cost for linear regression
    
    Arguments:
    - X,y: numpy.ndarray -- input data
    - theta: const 
    
    Returns
    - J: numpy.ndarray -- Computed cost.
    
    Raises:
    - None
    """
    # Initialize some useful values
    m = len(y) # number of training examples
    # J = np.zeros()
    J = np.sum((X.dot(theta.reshape((-1,1))) - y.reshape((-1,1)))**2, axis=0)/(2*m)
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Performs Gradient Descent to learn theta, updates theta by taking 
    num_iters gradient steps with learning rate alpha.
    
    Arguments:
    - X,y: input data
    - theta: initialize theta
    - alpha: learning rate
    - num_iters: number of iterations
    
    Raises:
    - None.
    
    Returns:
    - theta: np.ndarray -- updated theta
    - J_history: np.ndarray -- cost history through iterations
    """
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    
    for i in range(num_iters):
        delta = (1/m)* np.dot( X.T, np.dot(X, theta) - y.reshape((-1,1)))
        theta = theta - (alpha)*delta
        J_history[i] = compute_cost(X, y, theta)
    return (theta, J_history)    

def feature_normalize(X):
    """
    featureNormalize(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1.

    Arguments:
    - X: np.ndarray -- input features

    Raises:
    - None

    Returns:
    - X_norm: np.ndarray -- normalized features
    - mu : np.ndarray -- mean value of input features
    - sigma: np.ndarray -- standard devitation of input features
    
    """
    
    X_norm = X;
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    
    mu = np.mean(X, axis=0).reshape(mu.shape)
    sigma = np.std(X, axis=0, ddof=1).reshape(sigma.shape)

    mu_mat = np.tile(mu, (X.shape[0], 1))
    sigma_mat = np.tile(sigma, (X.shape[0], 1))              
    X_norm = (X_norm - mu_mat)/sigma_mat
    
    return (X_norm, mu, sigma)

def compute_cost_multi(X, y, theta):
    """
    Function to compute cost for linear regression

    Arguments:
    - X,y: np.ndarray -- input data
    - theta: const 

    Returns
    - J: np.ndarray -- Computed cost.

    Raises:
    - None
    """
    #Initialize some useful values
    m = len(y) # number of training examples
    a = (np.dot(X,theta) -y)**2
    J = np.sum(a, axis=0 )/(2*m)
    return float(J)

def gradient_descent_multi(X, y, theta, alpha, num_iters):
    """
    Performs Gradient Descent to learn theta, updates theta by 
    taking num_iters gradient steps with learning rate alpha.
    
    Arguments:
    - X,y: input data
    - theta: initialize theta
    - alpha: learning rate
    - num_iters: number of iterations
    
    Raises:
    - None.
    
    Returns:
    - theta: np.ndarray -- updated theta
    - J_history: np.ndarray -- cost history through iterations
    """
    m = y.shape[0]
    J_history = np.zeros((num_iters, 1))

    for iter in range(num_iters):
        delta = (np.dot(X.T, np.dot(X, theta) - y))/m
        theta -= alpha*delta
        J_history[iter] = compute_cost_multi(X, y, theta)

    return theta, J_history


def normal_eqn(X, y):
    """
    This function computes the closed-form solution to linear 
    regression using the normal equations.

    Arguments:
    - X, y: np.ndarray --- input data
    
    Raises:
    - None
    
    Returns:
    - theta: np.ndarray --- Computed theta
    """
    theta = np.zeros((X.shape[1],1))
    A = np.linalg.inv(np.dot(X.T,X))
    theta = np.dot(np.dot(A, X.T), y)
    return theta