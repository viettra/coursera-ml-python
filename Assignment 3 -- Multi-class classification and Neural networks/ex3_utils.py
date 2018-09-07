import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

__author__ = "VietTra"
__email__ = "viettra95@gmail.com"
__date__ = "08th_May_2018"

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

def display_data(X):
    [m,n] = X.shape
    example_width = int(np.round(np.sqrt(n)))
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # Between images padding
    pad = 1

    #  Setup blank display
    r = pad + display_rows*(example_height+pad)
    c = pad + display_cols*(example_width+pad)
    display_array = -np.ones((r,c))

    # Copy each example into a patch on the display array
    curr_ex = 1
    for j in range(1, display_rows+1):
        for i in range(1, display_cols+1):
            if curr_ex > (m):
                break
            # Copy the patch
            # Get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex-1,:]))
            dis_r = pad + (j - 1)*(example_height + pad)
            dis_c = pad + (i - 1)*(example_width + pad)
            rec = X[curr_ex-1,:].reshape((example_height, example_width)) / max_val
            display_array[dis_r:(dis_r + example_height), dis_c:(dis_c + example_width)] = rec.T 

            curr_ex += 1
            if curr_ex > (m):
                break
    return display_array

def lr_cost_function(theta, X, y, learning_rate):
    """
    This function computes cost and gradient for logistic regression with 
    regularization.
    
    Arguments:
    - theta, X, y: ndarray -- input data.
    - learningRate: scalar -- lambda.

    Raises:
    - None.
    
    Returns:
    - J: float -- computed cost
    - grad: ndarray -- computed gradient
    """
    m = len(y) # number of training examples
    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)
    h = sigmoid(np.dot(X, theta))

    #Compute cost:
    first = (1/m)*np.sum(np.dot(-y.T, np.log(h)) - np.dot((1-y).T, np.log(1-h)), axis=0)
    second = (learning_rate/(2*m))*np.sum(theta[1:]**2)
    J = first + second
    
    #Compute gradient
    grad = np.dot((h-y).T, X)/m
    grad = grad.T
    grad[1:] += (learning_rate/m)*theta[1:]
    return float(J), grad

class costFuncReg():
    """
    This class provides two function `cost` and `gradient` which compute cost and gradient
    for Regularized Logistic Regression
    """       
    def cost(theta, X, y, learning_rate):
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
        m = len(y)      
        theta = theta.reshape((-1,1))
        h = sigmoid(np.dot(X, theta))

        first = (1/m)*np.sum(np.dot(-y.T, np.log(h)) - np.dot((1-y).T, np.log(1-h)), axis=0)
        second = (learning_rate/(2*m))*np.sum(theta[1:]**2)
        J = first + second
        return float(J)
    
    def gradient(theta, X, y, learning_rate):
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
        m = len(y)

        theta = theta.reshape((-1,1))
        grad = np.zeros(theta.shape)
        h = sigmoid(np.dot(X, theta)) 
        
        grad = np.dot((h-y).T, X)/m
        grad = grad.T
        grad[1:] += (learning_rate/m)*theta[1:]
        return grad

def one_vs_all(X, y, num_labels, learning_rate):
    """
    This function  trains multiple logistic regression classifiers and returns all
    the classifiers in a matrix all_theta, where the i-th row of all_theta 
    corresponds to the classifier for label i.
    
    Arguments:
    - X,y: ndarray -- input date to train.
    - num_labels: int -- number of all labels.
    - learningRate: float -- lambda.
    
    Raises:
    - None.
    
    Returns:
    - all_theta: ndarray -- all the classifiers.
    """
    all_theta = np.zeros((10,401))
    m = len(y)
    X = np.append(np.ones((m,1)), X, axis=1)

    for i in range(1,num_labels+1): # labels are 1-indexed instead of 0-indexed
        theta = np.zeros((401,1))
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (5000,1))
        
        # minimize the objective function
        fmin = opt.minimize(fun=costFuncReg.cost, x0=theta, 
                            args=(X, y_i, learning_rate), 
                            method='TNC', jac=costFuncReg.gradient, 
                            tol=1e-8,
                            options={'maxiter':1000, 'disp':True})
        
        print('Cost %i:  '  %i +'%f' %costFuncReg.cost(fmin.x.reshape((-1,1)), X, y_i, learning_rate))
        all_theta[(i-1),:] = fmin.x
    return all_theta

def predict_one_vs_all(all_theta, X):
    """
    This function predicts the label for a trained one-vs-all classifier. 
    The labels are in the range 1..K, where K = all_theta.shape[0]. 
    
    Arguments:
    -all_theta: ndarray -- an array where the i-th row is a trained logistic
                          regression theta vector for the i-th class  .
    - X: ndarray -- input data.
    
    Raises:
    -None.
    
    Returns:
    - p: ndarray -- predictions for each example in the matrix X.
    """
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    X_add = np.append(np.ones((m,1)), X, axis=1)
    
    # compute the class probability for each class on each training instance
    h = sigmoid(np.dot(X_add, all_theta.T))    
    p = np.argmax(h, axis=1)
    
    # because our array was zero-indexed we need to add one for the true label prediction
    return p+1
    
def predict(theta1, theta2, X):
    """
    This function predicts the label of X given then trained weights 
    of a neural network (Theta1, Theta2).
    Arguments:
    - Theta1, Theta2: ndarray -- Trained weights of neural network.
    - X: ndarray -- Input data.

    Raises:
    - None.

    Returns:
    - p: ndarray -- predictions.
    """
    # Useful values
    m = X.shape[0]
    num_labels = theta2.shape[0]

    # compute the class probability for each class on each training instance
    a1 = np.append(np.ones((m,1)), X, axis = 1)
    a2 = np.append(np.ones((m,1)), sigmoid(np.dot(a1, theta1.T)), axis = 1)
    a3 = sigmoid(np.dot(a2, theta2.T))
    
    # because our array was zero-indexed we need to add one for the true label prediction
    p = np.argmax(a3, axis=1)
    return p+1