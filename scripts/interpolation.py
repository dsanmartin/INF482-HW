import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import distance_matrix

def plot(X, Y, Z, p='3D'):
    fig = plt.figure(figsize=(12,8))
    if p == '3D':
        ax = fig.gca(projection='3d')
        ax.scatter(X[:,0], X[:,1], Y)
        xx = np.linspace(0, 1, M)
        XX, YY = np.meshgrid(xx, xx)
        ax.plot_surface(XX, YY, Z)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
    else:
        plt.scatter(X[:,0], X[:,1], c=Y, marker='x')
        plt.imshow(Z, origin='lower', extent=[0,1,0,1])
    plt.show()

@jit(nopython=True)
def norm2(x):
    n = len(x)
    s = 0
    for i in range(n):
        s += x[i]**2   
    return np.sqrt(s)

@jit(nopython=True)
def distanceMatrix(X):
    N, s = X.shape
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = norm2(X[i] - X[j])        
    return D

def solveSystem(X, Y):
    D = distanceMatrix(X)
    C = np.linalg.solve(D, Y)
    return C
      
@jit(nopython=True)
def evaluatePol(X, x, C):
    N = len(X)
    p = 0
    for k in range(N):
        #print(norm2(x-X[k]))
        p += C[k]*norm2(x-X[k])
    return p 
    
    
#@jit(nopython=True)
def interpolate(X, Y, V):
    N = len(V)
    P = np.zeros(N)
    C = solveSystem(X, Y) 
    for i in range(N):
        P[i] = evaluatePol(X, V[i], C)  
    return P
        
#@jit(nopython=True)
def createV(M):
    V = []
    v = np.linspace(0, 1, M)
    for i in range(M):
        for j in range(M):
            V.append(np.array([v[i], v[j]]))            
    return np.array(V)

def main():    
    s = 2
    N = 5
    M = 50
    U = np.random.rand(N, s+1)
    X = U[:,:s]
    Y = U[:,-1]
    
    print(distanceMatrix(X))
    print(distance_matrix(X, X))
    
#    V = createV(M)
#    P = interpolate(X, Y, V)
#    plot(X, Y, P.reshape(M,M).T, p='2D')
    
    ## 1D
    #f = lambda x: np.sinc(x)
    #X = np.linspace(0, 1, N)
    #X = X.reshape(N, 1)
    #Y = f(X).reshape(N, 1)
    #V = np.linspace(0, 1, 100)
    #P = interpolate(X, Y, V)
    #plt.plot(X, Y, 'b*')
    #plt.plot(V, P)
    
main()
