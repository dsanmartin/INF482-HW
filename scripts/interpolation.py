import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import distance_matrix
from math import log, floor, ceil, fmod

def halton(dim, nbpts):
    h = np.empty(nbpts * dim)
    h.fill(np.nan)
    p = np.empty(nbpts)
    p.fill(np.nan)
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    lognbpts = log(nbpts + 1)
    for i in range(dim):
        b = P[i]
        n = int(ceil(lognbpts / log(b)))
        for t in range(n):
            p[t] = pow(b, -(t + 1) )

        for j in range(nbpts):
            d = j + 1
            sum_ = fmod(d, b) * p[0]
            for t in range(1, n):
                d = floor(d / b)
                sum_ += fmod(d, b) * p[t]

            h[j*dim + i] = sum_

    return h.reshape(nbpts, dim)

def plot(X, Y, Z, p='3D'):
    fig = plt.figure(figsize=(12,8))
    if p == '3D':
        ax = fig.gca(projection='3d')
        ax.scatter(X[:,0], X[:,1], Y, c='r')
        M = len(Z)
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
def testFunction(x):
    s = len(x)
    f = 4**s
    for d in range(s):
        f *= x[d] * (1 - x[d])
    return f

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

def eje1():
    pass
    #s = 2
    #N = 100
    #X = halton(s, N)
    #V = createV(N)
    #Z = np.zeros(len(V))
    #for i in range(len(V)):
    #    Z[i] = testFunction(V[i])        
    #plot(V, Z, Z.reshape(N, N).T, '3D')
    
    #f = lambda x, y: 
    
    
    

def eje2():    
    s = 2
    N = 50
    M = 100
    #U = np.random.rand(N, s+1)
    U = halton(3, N)
    X = U[:,:s]
    Y = U[:,-1]
    
    #print(distanceMatrix(X))
    #print(distance_matrix(X, X))
    
    V = createV(M)
    P = interpolate(X, Y, V)

    plot(X, Y, P.reshape(M, M).T, p='2D')
    #plot(X, Y, P.reshape(6, 6).T, p='3D')
    
    ## 1D
    #f = lambda x: np.sinc(x)
    #X = np.linspace(0, 1, N)
    #X = X.reshape(N, 1)
    #Y = f(X).reshape(N, 1)
    #V = np.linspace(0, 1, 100)
    #P = interpolate(X, Y, V)
    #plt.plot(X, Y, 'b*')
    #plt.plot(V, P)
    
eje2()
