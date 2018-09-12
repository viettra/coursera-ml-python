import numpy as np
import matplotlib.pyplot as plt
import time

__author__ = "VietTra"
__email__ = "viettra95@gmail.com"
__date__ = "17th_May_2018"

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example of dataset X.
    Arguments:
    - X: np.ndarray -- dataset.
    - centroids: np.ndarray
    
    Returns:
    idx: np.ndarray
    """
    m = X.shape[0]
    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly.
    idx = np.zeros((m, 1))
    
    #Code
    dis = np.zeros((K, 1))
    for i in range(m):
        for k in range(K):
            dis[k] = np.sum((X[i,:] - centroids[k, :])**2)
        idx[i] = np.argmin(dis)
    return idx+1

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the data points assigned to each centroid. 
    It is given a dataset X where each row is a single data point, a vector
    idx of centroid assignments (i.e. each entry in range [1..K]) for each
    example, and K, the number of centroids. You should return a matrix
    centroids, where each row of centroids is the mean of the data points
    assigned to it.
    """
    # Useful variables
    m, n = X.shape
    centroids = np.zeros((K, n))
    C = np.zeros((K, 1))
    for i in range(m):
        for k in range(K):
            if idx[i] == k+1:
                C[k] += 1
                centroids[k, :] += X[i, :]
    C = np.repeat(C, n, axis=1)
    centroids = centroids/C
    return centroids
                
def plot_progress_Kmeans(X, centroids, previous, idx, K, i, fig, ax):
    """
    Helper function that displays the progress of k-Means as it is running.
    It is intended for use only with 2D data.
    Plots the data points with colors assigned to each centroid. With the previous
        centroids, it also plots a line between the previous locations and
        current locations of the centroids.
    """
    # Plot the examples
    _ = plot_data_points(X, idx, K, ax)
    # Plot the centroids as black x's
    ax.plot(centroids[:,0], centroids[:,1], 'x', c='k', ms=10, lw=3)
    # Plot the history of the centroids with lines
    for j in range(centroids.shape[0]):
        ax.plot([centroids[j,0], previous[j,0]], 
                  [centroids[j,1],previous[j,1]], 'x-', c='k', ms=10, lw=1)
    plt.title('Iteration number {}'.format(i+1))
    fig.canvas.draw()
    return None

def plot_data_points(X, idx, K, ax):
    """
    # PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
    # index assignments in idx have the same color
    #    PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those 
    #    with the same index assignments in idx have the same color
    """
    cluster1 = X[np.where(idx == 1)[0],:]  
    cluster2 = X[np.where(idx == 2)[0],:]  
    cluster3 = X[np.where(idx == 3)[0],:]
    #fig=plt.figure(figsize=(4.5,3))

    ax.scatter(cluster1[:,0], cluster1[:,1], s=100, edgecolors='r', c='w', label='Cluster 1')  
    ax.scatter(cluster2[:,0], cluster2[:,1], s=100, edgecolors='g', c='w', label='Cluster 2')  
    ax.scatter(cluster3[:,0], cluster3[:,1], s=100, edgecolors='b', c='w', label='Cluster 3')  
    #plt.legend()  
    return None

def run_Kmeans(X, initial_centroids, max_iters, display=True, *args):
    """
    # %RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
    # %is a single example
    # %   [centroids, idx] = RUNKMEANS(X, initial_centroids, max_iters, ...
    # %   plot_progress) runs the K-Means algorithm on data matrix X, where each 
    # %   row of X is a single example. It uses initial_centroids used as the
    # %   initial centroids. max_iters specifies the total number of interactions 
    # %   of K-Means to execute. plot_progress is a true/false flag that 
    # %   indicates if the function should also plot its progress as the 
    # %   learning happens. This is set to false by default. runkMeans returns 
    # %   centroids, a Kxn matrix of the computed centroids and idx, a m x 1 
    # %   vector of centroid assignments (i.e. each entry in range [1..K])
    """
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))
    if display:
        if args:
            fig, ax = args
        else:
            fig, ax = plt.subplots()

    for i in range(max_iters):
        # Output progress
        print('K-Means iteration %i / %i...' %tuple((i+1, max_iters)))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        if display:
            _ = plot_progress_Kmeans(X, centroids, previous_centroids, idx, K, i, fig, ax)
        
        previous_centroids = centroids
        # Given the memberships, compute new centroids        
        centroids = compute_centroids(X, idx, K)
        time.sleep(.5)        
    return centroids, idx

def Kmeans_init_centroids(X, K):
    """
    # %KMEANSINITCENTROIDS This function initializes K centroids that are to be 
    # %used in K-Means on the dataset X
    # %   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
    # %   used with the K-Means on the dataset X
    """
    centroids = np.zeros((K, X.shape[1]))
    # Initialize the centroids to be random examples
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[0:K], :]
    return centroids

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

def pca(X):
    """
    Run principal component analysis on the dataset X
    """
    # Useful values
    m, n = X.shape

    # Compute the Covariance Matrix:
    sigma = (1/m)*np.dot(X.T, X)

    # #Principal components, Eigenvector, 
    U, S, V = np.linalg.svd(sigma)
    return U, np.diag(S)

def project_data(X, U, K):
    """
    Computes the reduced data representation when projecting only on to the top k eigenvectors.
    """
    # You need to return the following variables correctly.
    Z = np.zeros((X.shape[0], K))

    U_reduce = U[:, 0:K]
    for i in range(len(X)):
        Z[i, :] = np.dot(X[i, :], U_reduce)
    return Z

def recover_data(Z, U, K):
    """
    Recovers an approximation of the original data when using the projected data
    """
    X_rec = np.zeros((Z.shape[0], U.shape[0]))
             
    U_reduce = U[:, 0:K]
    for i in range(len(Z)):
        X_rec[i,:] = np.dot(Z[i, :], U_reduce.T)
    return X_rec

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