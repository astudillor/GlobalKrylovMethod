# Copyright (c) 2015 Reinaldo Astudillo and Martin B. van Gijzen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import numpy as np
from numpy import vdot as dot
from numpy.linalg import norm
from numpy.linalg import solve

__all__ = ['idrs']

def idrs(A, B, X0=None, tol=1e-8, s=4, maxit=None, M1=None, callback=None):
    """
    idrs - solves the linear matrix equation A(X) = B using IDR(s) method
    """
    size = B.shape
#   Size of the residuals
    if maxit is None:
        maxit = 2*np.prod(size)
    if M1 is None:
#   No preconditioner class
        class __NoPrecond__(object):
            def solve(self,_X_): return _X_
        M1 = __NoPrecond__()
    if X0 is None:
#   Default initial guess
        xtype = np.result_type(A, B)
        X0 = np.zeros(size, dtype = xtype)
    else:
        xtype = np.result_type(A, B, X0)

    X = np.array(X0)
    bnrm = norm(B)
    info = 1

#   Check for zero rhs:    
    if bnrm == 0.0:
#   Solution is null-vector
        info = 0
        return np.zeros(size), info
#   Compute initial residual:
    R = B - A.dot(X)
    rnrm = norm(R)
#   Relative tolerance
    tolb = tol*bnrm
    if callback is not None:    
        callback(R)

    if rnrm < tolb:
# Initial guess is a good enough solution
        info = 0
        return X, info

#   Initialization
    np.random.seed(0)
    P = [np.random.random_sample(size)  for i in range(0,s)]
    angle = 0.78539816339
    G = [np.zeros(size,dtype=xtype) for i in range(0,s)]
    U = [np.zeros(size,dtype=xtype) for i in range(0,s)]
    Q = np.zeros(size,dtype=xtype)
    V = np.zeros(size,dtype=xtype)
    M = np.eye(s,dtype=xtype)
    f = np.zeros([s],dtype=xtype)
    om = 1.0
    iter = 0

#   Main iteration loop, build G-spaces:
    while rnrm > tolb and iter < maxit:
#   New righ-hand size for small system:
        for i in range(0,s):
            f[i] = dot(P[i], R)

        for k in range(0,s):
#       Solve small system and make v orthogonal to P:
            c = solve(M[k:s,k:s],f[k:s])
            V = c[0]*G[k]
            Q = c[0]*U[k]
            for i in range(k+1,s):
                V += c[i-k]*G[i]
                Q += c[i-k]*U[i]
            V = R-V 
#           Preconditioning:
            V = M1.solve(V)
            U[k] = Q + om*V
#           Compute new U(:,k) and G(:,k), G(:,k) is in space G_j
#           Application of the linear operator    
            G[k] = A.dot(U[k])

#           Bi-Orthogonalise the new basis vectors: 
            for i in range(0,k):
                alpha = dot(P[i],G[k])/M[i,i]
                G[k] -= alpha*G[i]
                U[k] -= alpha*U[i]
#           New column of M = P'*G  (first k-1 entries are zero)
            for i in range(k,s):
                M[i,k] = dot(P[i],G[k])

#           Make r orthogonal to q_i, i = 1..k             
            beta = f[k]/M[k,k]
            X +=  beta*U[k]
            R -=  beta*G[k]
            rnrm = norm(R)
            if callback is not None:    
                callback(R)
            iter += 1
            if rnrm < tolb or iter >= maxit: 
                info = 0
                break
            if k < s-1:
                f[k+1:s] = f[k+1:s] - beta*M[k+1:s,k]
# Now we have sufficient vectors in G_j to compute residual in G_j+1
# Note: R is already perpendicular to P so v = r
        if rnrm < tolb or iter >= maxit: 
            info = 0
            break
#       Preconditioning:
        V = M1.solve(R)
#       Application of the linear operator    
        Q = A.dot(V)
#       Computation of a new omega
####################################
        ns = norm(R)
        nt = norm(Q)
        ts = dot(Q,R) 
        rho = np.abs(ts/(nt*ns))
        om = ts/(nt*nt)
        if rho < angle:
            om = om*angle/rho
####################################    

        X += om*V
        R -= om*Q
        rnrm = norm(R)
        if callback is not None:    
            callback(R)
        iter += 1

    if rnrm < tolb:   info = 0
    return X, info
