import numpy as np
import matplotlib.pyplot as plt
from ex5_utils import *
import scipy.io as io
import scipy.optimize as opt
from IPython.display import *
from sklearn.preprocessing import OneHotEncoder

__author__ = "VietTra"
__email__ = "viettra95@gmail.com"
__date__ = "11st_May_2018"

class linearRegCostFunction(object):
    """
    This class provides two function `cost` and `gradient` which compute cost and gradient
    for Logistic Regression
    """
    def __init__(self, X=None, y=None, lb=None):
        assert X is not None, \
            "The argument `X` must be specified!"
        assert y is not None, \
            "The argument `y` must be specified!" 
        self.X = X
        self.y = y
        self.lb = lb
        self.m = len(y)
        
    def cost(self, theta):
        """
        Compute cost and gradient for regularized linear regression with multiple variables
        Returns the cost in J and the gradient in grad

        Arguments:
        - theta: np.ndarray

        Returns:
        - J, grad
        """ 
        X = self.X
        y = self.y
        lb = self.lb
        m = self.m

        nonReg = np.sum((np.dot(X, theta) - y)**2) / (2*m)
        reg = (lb/(2*m)) * np.sum(theta[1:]**2)
        J = nonReg + reg
        return J

    def gradient(self, theta):

        X = self.X
        y = self.y
        lb = self.lb
        m = self.m
        
        y_pred = np.dot(X, theta)    
        grad = np.dot(X.T, y_pred-y)/m
    
        grad[1:] += theta[1:]/m
        return grad.reshape((-1,))
        
def train_linear_reg(X, y, lb):
    """
    Trains linear regression given a dataset (X, y) and a regularization parameter lambda
    Returns the trained parameters theta.
    """
    # Initialize Theta
    initial_theta = np.zeros((X.shape[1], 1)) 
    theta = fmincg(initial_theta, X, y, lb)
    return theta
    
def learning_curve(X, y, Xval, yval, lb):
    """
    Generates the train and cross validation set errors needed plot a learning curve
    In particular, it returns two vectors of the same length - error_train and error_val.
    Then, error_train[i] contains the training error for i examples (and similarly for error_val[i]).
    """
    # Number of training examples
    m = X.shape[0]

    error_train = np.zeros((m,1))
    error_val = np.zeros((m,1))
    for i in range(m):
        #Using different subsets of training size:
        Xsub = X[0:(i+1),:]
        ysub = y[0:(i+1)]
        
        #Compute train/CV errors
        theta = train_linear_reg(Xsub, ysub, lb)  
        error_train[i] = linearRegCostFunction(Xsub, ysub, 0).cost(theta)
        error_val[i] = linearRegCostFunction(Xval, yval, 0).cost(theta)
    return error_train, error_val

def poly_features(X, p):
    """
     Maps X (1D vector) into the p-th power
     Returns a maxtrix X_poly where the p-th comlun of X contains the values of X to the p-th power.
    """
    X_poly = np.zeros((X.shape[0]*X.shape[1], p))
    m = X.shape[0]
    
    for i in range(m):
        poly_feature = np.zeros((p, 1))
        for j in range(p):
            poly_feature[j] = X[i,0]**(j+1)
        #print('X_poly shape', X_poly.shape)
        X_poly[i,:] = poly_feature.reshape(-1)
    return X_poly

def feature_normalize(X):
    """
    Normalizes the features in X
    Returns a normalized version of X where the mean value of each feature
    is 0 and the standard deviationis 1. 
    This is often a good preprocessing step to do when working with learning algorithms.
    """
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    
    sigma = np.std(X_norm, ddof=1, axis=0)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma

def plotFit(min_x, max_x, mu, sigma, theta, p):
    """
    Plots a learned polynomial regression fit over an existing figure.
    Also works with linear regression.
    """
    x = np.arange(min_x-15, max_x+25, 0.05)
    x = x.reshape((-1,1))
    # x = (min_x - 15: 0.05 : max_x + 25)';

    #  Map the X values 
    X_poly = poly_features(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma

    # Add ones
    X_poly = np.append(np.ones((x.shape[0], 1)), X_poly, axis=1)

    # Plot
    plt.plot(x, np.dot(X_poly, theta), '--', lw=2)
    return None

def fmincg(theta, *args):
    """Custom func for training"""
    theta_reshape = np.reshape(theta, (-1))
    X, y, lb = args
    
    def cost_mod(theta, *args):
        theta = theta.reshape((-1,1))
        X, y, lb = args
        J = linearRegCostFunction(X, y, lb).cost(theta)
        return float(J)
    def grad_mod(theta, *args):
        theta = theta.reshape((-1,1))
        X, y, lb = args
        grad = linearRegCostFunction(X, y, lb).gradient(theta)
        return grad.reshape((-1))

    theta_opt = opt.fmin_cg(f=cost_mod, x0=theta_reshape, fprime=grad_mod,
                              args=(X, y, lb), maxiter=200, disp=0)
    
    return np.reshape(theta_opt,(-1,1))

def validation_curve(X, y, Xval, yval):
    """
    Generate the train and validation errors needed to plot a validation curve 
    that we can use to select lambda
    """
    # Selected values of lambda
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    lambda_vec = lambda_vec.reshape((-1,1))
    
    error_train = np.zeros_like(lambda_vec)
    error_val = np.zeros_like(lambda_vec)
    for i in range(len(lambda_vec)):
        lamb = lambda_vec[i]
        
        #Compute train/CV errors:
        theta = train_linear_reg(X, y, lamb)
        error_train[i] = linearRegCostFunction(X, y, 0).cost(theta)
        error_val[i] = linearRegCostFunction(Xval, yval, 0).cost(theta)
    
    return lambda_vec, error_train, error_val