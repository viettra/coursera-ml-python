import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

__author__ = "VietTra"
__email__ = "viettra95@gmail.com"
__date__ = "18th_May_2018"

def estimate_gaussian(X):
    """
    This function estimates the parameters of a Gaussian distribution using the data in X
    Arguments: 
    - X: np.ndarray -- dataset.
    
    Raises: None.
    
    Returns:
    - mu: np.ndarray -- mean of the dataset.
    - sigma2: np.ndarray: -- the variances vector.
    """
    # Useful variables
    m, n = X.shape

    mu = np.mean(X, axis=0).reshape((1,-1))
    sigma2 = np.repeat(mu, m, axis=0)
    sigma2 = np.sum((X - sigma2)**2, axis=0)/m
    sigma2 = np.reshape(sigma2, (1,-1))
    return mu, sigma2

def multivariate_gaussian(X, mu, Sigma2):
    """
    # %MULTIVARIATEGAUSSIAN Computes the probability density function of the
    # %multivariate gaussian distribution.
    # %    p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability 
    # %    density function of the examples X under the multivariate gaussian 
    # %    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
    # %    treated as the covariance matrix. If Sigma2 is a vector, it is treated
    # %    as the \sigma^2 values of the variances in each dimension (a diagonal
    # %    covariance matrix)
    """
    k = mu.shape[1]

    if Sigma2.shape[0] == 1 or Sigma2.shape[1] == 1:
        Sigma2 = np.diag(Sigma2.ravel())

    X = X - mu
    p1 = (2*np.pi)**(-k/2) * np.linalg.det(Sigma2)**(-0.5) 
    p2 = np.exp(-0.5*np.sum(np.dot(X, np.linalg.pinv(Sigma2))*X, axis=1))
    return (p1*p2).reshape((-1,1))

def visualize_fit(X, mu, sigma2):
    """
    # %VISUALIZEFIT Visualize the dataset and its estimated distribution.
    # %   VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you the 
    # %   probability density function of the Gaussian distribution. Each example
    # %   has a location (x1, x2) that depends on its feature values.
    """
    X1, X2 = np.mgrid[0:35.5:.5, 0:35.5:.5]
    X12 = np.append(np.reshape(X1, (-1,1)), np.reshape(X2, (-1,1)), axis=1)
    Z = multivariate_gaussian(X12, mu, sigma2)
    Z = np.reshape(Z, X1.shape)

    plt.figure(figsize=(6,4))
    plt.plot(X[:,0], X[:, 1], 'bx', ms=5)

    # Do not plot if there are infinities
    if np.sum(np.isinf(Z)) == 0:
        a = (10*np.ones((7,1)))**(np.arange(-20, 0, 3).reshape((-1,1)))
        plt.contour(X1, X2, Z, a)

    plt.axis([0,30,0,30])
    return None

def select_threshold(yval, pval):
    """
    # %SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
    # %outliers
    # %   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
    # %   threshold to use for selecting outliers based on the results from a
    # %   validation set (pval) and the ground truth (yval).
    """
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    stepsize = (np.max(pval) - np.min(pval)) / 1000
    c = 0
    for epsilon in np.arange(np.min(pval), np.max(pval)+stepsize, stepsize):
        predictions = (pval < epsilon)
        fp = np.sum(np.logical_and((predictions==1), (yval==0)))
        tp = np.sum(np.logical_and((predictions==1), (yval==1)))
        fn = np.sum(np.logical_and((predictions==0), (yval==1)))

        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        F1 = (2*prec*rec)/(prec + rec)

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1

def cofi_cost_func(params, Y, R, num_users, num_movies, num_features, lamb):
    """
    # %COFICOSTFUNC Collaborative filtering cost function
    # %   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
    # %   num_features, lambda) returns the cost and gradient for the
    # %   collaborative filtering problem.
    """
    
    # Unfold the U and W matrices from params
    X = np.reshape(params[0:num_movies*num_features], (num_movies, num_features), order='F')
    Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features), order='F')

    # You need to return the following values correctly
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    reg = (lamb/2) * (np.sum(np.sum(Theta**2)) + np.sum(np.sum(X**2)))

    J = np.sum(np.sum(R*(np.dot(X, Theta.T)-Y)**2))/2 + reg

    X_grad = np.dot(R*(np.dot(X, Theta.T)-Y), Theta) + lamb*X

    Theta_grad = np.dot((R*(np.dot(X, Theta.T)-Y)).T, X) + lamb*Theta

    grad = np.append(X_grad.T.ravel(), Theta_grad.T.ravel())
    return J, np.reshape(grad, (-1,1))

def compute_numerical_gradient(*args, theta):
    #Y, R, num_users, num_movies, num_features, lamb
    """
    # function numgrad = computeNumericalGradient(J, theta)
    # %COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
    # %and gives us a numerical estimate of the gradient.
    # %   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
    # %   gradient of the function J around theta. Calling y = J(theta) should
    # %   return the function value at theta.
    """
    ### All arguments:
    Y, R, num_users, num_movies, num_features, lamb = args

    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    
    for p in range(np.prod(theta.shape)):
        #Set perturbation vector
        perturb[p] = e
        loss1 = cofi_cost_func(theta - perturb, Y, R, num_users,
                               num_movies, num_features, lamb)[0]
        loss2 = cofi_cost_func(theta + perturb, Y, R, num_users,
                               num_movies, num_features, lamb)[0]
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
    return numgrad.reshape((-1,1))


def check_cost_function(lamb=None):
    """
    # %CHECKCOSTFUNCTION Creates a collaborative filering problem 
    # %to check your cost function and gradients
    # %   CHECKCOSTFUNCTION(lambda) Creates a collaborative filering problem 
    # %   to check your cost function and gradients, it will output the 
    # %   analytical gradients produced by your code and the numerical gradients 
    # %   (computed using computeNumericalGradient). These two gradient 
    # %   computations should result in very similar values.
    """
    # Set lambda
    if lamb is None:
        lamb = 0


    lamb = 1
    # Create small problem
    X_t = np.random.rand(4,3)
    Theta_t = np.random.rand(5,3)

    # Zap out most entries
    Y = np.dot(X_t, Theta_t.T)
    Y[np.random.rand(Y.shape[0], Y.shape[1]) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    # Run Gradient Checking
    X = np.random.randn(X_t.shape[0], X_t.shape[1])
    Theta = np.random.randn(Theta_t.shape[0], Theta_t.shape[1])
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

    params = np.append(X.T.ravel(), Theta.T.ravel())

  
    thetaArg = np.append(X.T.ravel(), Theta.T.ravel())
    numgrad = compute_numerical_gradient(Y, R, num_users, num_movies,
                                       num_features, lamb, theta=thetaArg)
    cost, grad = cofi_cost_func(thetaArg, Y, R, num_users, num_movies, num_features, lamb)

    disp = np.append(numgrad, grad, axis=1)
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('If your cost function implementation is correct, then \n' \
         'the relative difference will be small (less than 1e-9). \n'\
         'Relative Difference: %e' %diff)
    print('The below two columns you get should be very similar.\n' \
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')
    print(disp)
    return None


def load_movie_list():
    # function movieList = loadMovieList()
    # %GETMOVIELIST reads the fixed movie list in movie.txt and returns a
    # %cell array of the words
    # %   movieList = GETMOVIELIST() reads the fixed movie list in movie.txt 
    # %   and returns a cell array of the words in movieList.

    movie_idx = {}  
    # %% Read the fixed movieulary list
    f = open('ex8_data/movie_ids.txt')  


    # % Store all movies in cell array movie{}
    for line in f:  
        tokens = line.split(' ')
        tokens[-1] = tokens[-1][:-1] # delete \n letter
        movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])
    return movie_idx


def normalize_ratings(Y, R):
    """
    # %NORMALIZERATINGS Preprocess data by subtracting mean rating for every 
    # %movie (every row)
    # %   [Ynorm, Ymean] = NORMALIZERATINGS(Y, R) normalized Y so that each movie
    # %   has a rating of 0 on average, and returns the mean rating in Ymean.
    """
    
    m, n = Y.shape
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros(Y.shape)
    for i in range(m):
        idx = np.where(R[i,:] == 1)
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
        
    return Ynorm, Ymean

def fmincg(theta, *args):
    Y, R, num_users, num_movies, num_features, lamb = args
    theta_reshape = np.reshape(theta, (-1))

    def cost_mod(theta, *args):
        theta = theta.reshape((-1,1))
        Y, R, num_users, num_movies, num_features, lamb = args
        J = cofi_cost_func(theta, Y, R, num_users, num_movies, num_features, lamb)[0]
        return float(J)
    def grad_mod(theta, *args):
        theta = theta.reshape((-1,1))
        Y, R, num_users, num_movies, num_features, lamb = args
        grad = cofi_cost_func(theta, Y, R, num_users, num_movies, num_features, lamb)[1]
        return grad.reshape((-1))
    
    theta_opt = opt.fmin_cg(f=cost_mod, x0=theta_reshape, fprime=grad_mod,
                              args=( Y, R, num_users, num_movies, num_features, lamb)
                              ,maxiter=50, disp=True, full_output=True)
    
    #result = opt.fmin_cg(cofiCostFunc, x0=myflat, fprime=cofiGrad, args=(Y,R,nu,nm,nf,mylambda),
     #                              maxiter=50,disp=True,full_output=True)
    return np.reshape(theta_opt,(-1,1))