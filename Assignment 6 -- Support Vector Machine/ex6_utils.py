import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import scipy.optimize as opt
from IPython.display import *
from sklearn import svm
import os 

__author__ = "VietTra"
__email__ = "viettra95@gmail.com"
__date__ = "15th_May_2018"

def plotData(X, y):
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
    - fig : matplotlib.figure
    """
       
    fig = plt.figure()
    
    neg = np.where(y==0)[0].reshape((-1,1))
    pos = np.where(y==1)[0].reshape((-1,1))
    
    plt.plot(X[pos,0], X[pos,1], 'k+', lw=2, ms=7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko',mfc='y', ms=7)
    return fig

def visualizeBoundaryLinear(X, y, model):
    """
    # %VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
    # %SVM
    # %   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary 
    # %   learned by the SVM and overlays the data on it
    """

    w = model.coef_
    b = model.intercept_
    xp = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    yp = - (w[0,0]*xp + b)/w[0,1]
    _ = plotData(X, y)
    plt. plot(xp, yp, '-b')
    return None

def gaussianKernel(x1, x2, sigma):
    """
    # This function returns a radial basis function kernel between x1 and x2
    # sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
    # and returns the value in sim
    """

    # Ensure that x1 and x2 are column vectors
    x1 = x1.reshape((-1,1))
    x2 = x2.reshape((-1,1))
    
    sim = np.exp((-1/(2*(sigma**2))) * np.sum((x1-x2)**2))
    return sim
    
def visualizeBoundary(X, y, model):
    """
    # %VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
    # %   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
    # %   boundary learned by the SVM and overlays the data on it
    """



    # Plot the training data on top of the boundary
    plotData(X, y)
    

    # Make classification predictions over a grid of values
    x1plot = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100).reshape((-1,1))
    x2plot = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100).reshape((-1,1))
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in range(len(X1)):
        this_X = np.append(X1[:,i:i+1], X2[:,i:i+1], axis=1)
        vals[:,i] = model.predict(this_X)
        
    plt.contour(X1, X2, vals, levels=[0.5, 1], colors='b')

    return None

def dataset3Params(X, y, Xval, yval):
    """
    # function [C, sigma] = dataset3Params(X, y, Xval, yval)
    # %DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
    # %where you select the optimal (C, sigma) learning parameters to use for SVM
    # %with RBF kernel
    """
    values = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    results = np.zeros((len(values)**2, 3))

    for i in range(len(values)):
        for j in range(len(values)):
            testC = values[i]
            testSigma = values[j]

            model = svm.SVC(C=testC, gamma=testSigma, probability=True)
            model.fit(X, y.ravel())
            predMat = model.predict(Xval)
            predictions = np.not_equal(predMat.reshape((-1,1)), yval)
            testError = np.mean(predictions)
            results[i*8 + j, :] = np.append(np.append(testC.reshape((-1,1)),
                                              testSigma.reshape((-1,1)), axis=1),
                                               testError.reshape((-1,1)), axis=1)
            minIndex = np.argmin(results[:,2])
            C = results[minIndex, 0]
            sigma = results[minIndex,1]
    return C, sigma

def getVocabList():
    """ This function reads the fixed vocabulary list in vocab.txt and returns a
    list of the words.
    
    Arguments:
    - None
    
    Returns:
    - vocabList: dict -- including words in the `vocab.txt`
    """
    
    vocabList = {}
    with open("vocab.txt") as f:
        for line in f:
           (key, val) = line.split()
           vocabList[int(key)] = val
    return vocabList

def readFile(filename):
    """ This function reads a file and returns its entire contents.
    
    Arguments:
    - filename: str -- path location of the file
    
    Returns:
    file_contents: list -- including entire contents (characters) from the file
    """
    
    if os.path.exists(filename) == False:
        print('Unable to open {}'.format(filename))
        file_contents = ""
    else:
        f = open(filename)
        file_contents = f.read()
        f.close
    return file_contents