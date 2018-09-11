import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import scipy.optimize as opt
from IPython.display import *
from IPython.display import *
from sklearn.preprocessing import OneHotEncoder 

__author__ = "VietTra"
__email__ = "viettra95@gmail.com"
__date__ = "08th_May_2018"

def sigmoid(z):
    """
    Function sigmoid(z) computes the sigmoid of z.
        Arguments:
    - z: input value to compute, can be a array, vector or scalar.
    
    Raises:
    - None.
    
    Returns:
    - g: computed sigmoid value.
    """
    g = 1/(1 + np.exp(-z))
    return g

def sigmoid_gradient(z):
    """
    This function computes the the gradient of the sigmoid function evaluated at z.
    Arguments:
    - z: input value to compute, can be a array, vector or scalar.
    
    Raises:
    - None.
    
    Returns:
    - g: ndarray -- computed gradient for sigmoid function.
    """
    g = sigmoid(z)*(1 - sigmoid(z))
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

def forward_propagate(X, Theta1, Theta2):  
    m = X.shape[0]
    
    a1 = np.append(np.ones((m, 1)), X, axis=1)    
    z2 = np.dot(a1, Theta1.T)
    
    a2 = np.append(np.ones((m, 1)), sigmoid(z2), axis=1)
    z3 = np.dot(a2, Theta2.T)

    h = sigmoid(z3)
    return a1, z2, a2, z3, h

def compute_cost(nn_params,input_layer_size, hidden_layer_size, num_labels, X, y, lb):
    # Reshape nn_params back into the parameters Theta1 and Theta2,
    Theta1 = nn_params[:hidden_layer_size*(input_layer_size+1)]
    Theta1 = np.reshape(Theta1, (hidden_layer_size, input_layer_size+1), order='F')
    
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):]
    Theta2 = np.reshape(Theta2, (num_labels, hidden_layer_size+1), order='F')
    
    # Setup some useful variables
    m = X.shape[0]
    delta1 = np.zeros(Theta1.shape)  # (25, 401)
    delta2 = np.zeros(Theta2.shape)  # (10, 26)
    
    ##############
    a1, z2, a2, z3, h = forward_propagate(X, Theta1, Theta2)
    encoder = OneHotEncoder(sparse=False)  
    y_onehot = encoder.fit_transform(y) 
    
    # Feedforward the neural network and return the cost in the variable J 
    J = -y_onehot*np.log(h) - (1 - y_onehot)*np.log(1 - h)
    J = (1/m)*np.sum(np.sum(J))
    
    # add the cost regularization term
    reg1 = np.sum(np.sum(Theta1[:,1:]**2))
    reg2 = np.sum(np.sum(Theta2[:,1:]**2))
    J = J + (lb/(2*m))*(reg1+reg2)
    return J

def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lb):
    """
    This function implements the neural network cost function for a two layer
    neural network which performs classification.
    
    The parameters for the neural network are "unrolled" into the vector
        `nn_params` and need to be converted back into the weight matrices. 
    
    Arguments:
    - nn_params: ndarray -- "unrolled" parameters for the neural network.
    - input_layer_size, hidden_layer_size, num_labels: int.
    - X,y: ndarray -- input data.
    - lb: float -- input lambda/ learning rate.
    
    Raises:
    - None.
    
    Returns:
    - J: float -- Computed cost.
    - grad: ndarray -- Computed gradient of hidden layer.
    """
    J = compute_cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lb)
    grad = (nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lb)
    return J, grad

def compute_gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lb):
    
    # Reshape nn_params back into the parameters Theta1 and Theta2,
    Theta1 = nn_params[:hidden_layer_size*(input_layer_size+1)]
    Theta1 = np.reshape(Theta1, (hidden_layer_size, input_layer_size+1), order='F')
    
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):]
    Theta2 = np.reshape(Theta2, (num_labels, hidden_layer_size+1), order='F')

    # Setup some useful variables
    m = X.shape[0]
    delta1 = np.zeros(Theta1.shape)  # (25, 401)
    delta2 = np.zeros(Theta2.shape)  # (10, 26)
    
    a1, z2, a2, z3, h = forward_propagate(X, Theta1, Theta2)
    encoder = OneHotEncoder(sparse=False)  
    y_onehot = encoder.fit_transform(y) 
    
    for t in range(m):
        ht = h[t:t+1,:].T        # (10, 1)
        a1t = a1[t:t+1,:].T      # (401, 1)
        a2t = a2[t:t+1,:].T      # (26, 1)
        yt = y_onehot[t:t+1,:].T # (10, 1)

        d3t = ht - yt # (10, 1)
        z2t = np.append(np.ones((1,1)), np.dot(Theta1, a1t), axis=0) # (26, 1)

        d2t = np.dot(Theta2.T, d3t)*(sigmoid_gradient(z2t)) # (26, 1)

        delta1 = delta1 + np.dot(d2t[1:], a1t.T)
        delta2 = delta2 + np.dot(d3t, a2t.T)

    Theta1_grad = (1/m)*delta1
    Theta2_grad = (1/m)*delta2

    Theta1Zeros = np.append(np.zeros((hidden_layer_size, 1)), Theta1[:, 1:], axis=1)
    Theta2Zeros = np.append(np.zeros((num_labels, 1)), Theta2[:, 1:], axis=1)

    Theta1_grad = (1/m)*delta1 + (lb/m)*Theta1Zeros
    Theta2_grad = (1/m)*delta2 + (lb/m)*Theta2Zeros
    
    grad = np.append(Theta1_grad.flatten(order='F'), Theta2_grad.flatten(order='F'), axis=0)
    grad = grad.reshape((-1,1))
    return grad
    
def rand_initialize_weights(L_in, L_out):
    """
    This function randomly initialize the weights of a layer with 
        L_in incoming connections and L_out outgoing connections.
    
    Arguments:
    -
    
    Raises:
    - None.
    
    Returns:
    - W -- ndarray: The first column of W corresponds to the parameters 
                    for the bias unit
    """
    
    W = np.zeros((L_out, 1+L_in))
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W

def compute_numerical_gradient(J, theta):
    """
    Computes the gradient using "finite differences" and gives us a numerical estimate of the gradient.
    """
    
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    
    e = 10**(-4)
    for i in range (theta.shape[0]):
        for j in range(theta.shape[1]):
            #Set perturbation vector
            perturb[i,j] = e
            loss1 = J(theta - perturb)
            loss2 = J(theta + perturb)
            #Compute Numerical Gradient
            numgrad[i,j] = (loss2 - loss1) / (2*e)
            perturb[i,j] = 0
    return numgrad

def _debug_initialize_weights(fan_out, fan_in):
    
    """
    Initialize the weights of a layer with `fan_in` incoming connections and `fan_out` outgoing connections 
    using a fixed strategy, this will help you later in debugging.
    
    Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
    the first row of W handles the "bias" terms
    """
    # Set W to zeros
    W = np.zeros((fan_out, 1 + fan_in))

    # Initialize W using "sin", this ensures that W is always of the same values and will be useful for debugging
    S = np.sin(np.arange(1, W.shape[0]*W.shape[1] +1, 1))/10
    W = np.reshape(S, W.T.shape)
    return W.T

def check_nn_gradients(lb=0):
    """
    Creates a small neural network to check the backpropagation gradients,it will output the 
    analytical gradients produced by your backprop code and the numerical gradients (computed
    using compute_numerical_gradient). These two gradient computations should result 
    in very similar values.
    """
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = _debug_initialize_weights(hidden_layer_size, input_layer_size)
    Theta2 = _debug_initialize_weights(num_labels, hidden_layer_size)
    
    # Reusing debugInitializeWeights to generate X
    X = _debug_initialize_weights(m, input_layer_size - 1)
    y = 1 + np.remainder(np.arange(1,m+1,1), num_labels).reshape((-1,1))
    
    # Unroll parameters 
    nn_params = np.append(Theta1.flatten(order='K'), Theta2.flatten(order='K'), axis=0)
    nn_params = nn_params.reshape((-1,1))

    cost = lambda p: compute_cost(p, input_layer_size, hidden_layer_size, num_labels, X, y, lb)
    grad = compute_gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lb)
    numgrad = compute_numerical_gradient(cost, nn_params)
    
    # Visually examine the two gradient computations. The two columns you get should be very similar.
    print('The below two columns you get should be very similar')
    print('Left-Your Numerical Gradient, Right-Analytical Gradient)')
    print(np.append(numgrad, grad, axis=1))

    # Evaluate the norm of the difference between two solutions.  
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad-grad)
    diff /= np.linalg.norm(numgrad+grad)
    
    print('If your backpropagation implementation is correct, then the'
          'relative difference will be small (less than 1e-9).')
    print('Relative Difference: %g' %diff)
    return None

def predict(Theta1, Theta2, X):
    """
    Predict the label of an input given a trained neural network. 
    Arguments:
    - Theta1, Theta2: np.ndarray -- Trained weights of a neural network.
    - X: np.ndarray -- Input data to predict label.
    
    Returns: p -- np.ndarray -- Predicted label.
    """
    
    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros((m, 1))
    term1 = np.append(np.ones((m, 1)), X, axis=1)
    h1 = sigmoid(np.dot(term1,Theta1.T))
    
    term2 = np.append(np.ones((m,1)), h1, axis=1)
    h2 = sigmoid(np.dot(term2,Theta2.T))
    p = np.argmax(h2, axis=1)
    return p+1