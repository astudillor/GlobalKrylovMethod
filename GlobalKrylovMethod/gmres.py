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
from scipy.linalg.blas import get_blas_funcs

__all__ = ['gmres']

def gmres(A, B, m=200, X0=None, tol=1e-8, maxit=None, M1=None, callback=None):
    """
    gmres - solves the linear matrix equation A(X) = B using global GMRES(m) method
    """
    size = B.shape
    # Size of the residuals
    if maxit is None:
        maxit = 2*np.prod(size)
    if M1 is None:
        # No preconditioner class
        class __NoPrecond__(object):
            def solve(self,_X_): return _X_
        M1 = __NoPrecond__()
    if X0 is None:
    # Default initial guess
        xtype = np.result_type(A, B)
        X0 = np.zeros(size, dtype = xtype)
    else:
        xtype = np.result_type(A, B, X0)

    X = np.array(X0)
    bnrm = norm(B)
    info = 1

    # Check for zero rhs:    
    if bnrm == 0.0:
        # Solution is null-vector
        info = 0
        return np.zeros(size), info
    # Compute initial residual:
    R = M1.solve(B - A.dot(X))
    rnrm = norm(R)
    # Relative tolerance
    tolb = tol*bnrm
    if callback is not None:    
        callback(rnrm)

    if rnrm < tolb:
        # Initial guess is a good enough solution
        info = 0
        return X, info
    
    # Initialization
    rotmat = get_blas_funcs('rotg', dtype=xtype)
    V = [np.zeros(size, dtype=xtype) for i in range(0, m+1)]
    H = np.zeros((m+1, m), dtype=xtype)
    cs = np.zeros(m+1, dtype=xtype)
    sn = np.zeros(m+1, dtype=xtype)
    e1 = np.zeros(m+1, dtype=xtype)
    e1[0] = 1.0
    for _iter in range(0, maxit):
        # Begin iteration
        V[0] = R/rnrm
        s = rnrm*e1
        for i in range(0, m):
            # Construct orthonormal basis
            # using Gram-Schmidt
            W = M1.solve(A.dot(V[i]))
            for k in range(0, i):
                H[k, i] = dot(W, V[k])
                W = W - H[k, i]*V[k]
                print(dot(W, V[k]))
            H[i+1, i] = norm(W)
            V[i+1] = W/H[i+1, i]
            for k in range(0, i-1):
                # Apply Givens rotation
                temp = cs[k]*H[k, i] + sn[k]*H[k+1, i]
                H[k+1, i] = -sn[k]*H[k, i] + cs[k]*H[k+1, i]
                H[k, i] = temp
            cs[i], sn[i] = rotmat(H[i, i], H[i+1, i])
            temp   = cs[i]*s[i]
            s[i+1] = -sn[i]*s[i]
            s[i] = temp
            H[i, i] = cs[i]*H[i, i] + sn[i]*H[i+1, i]
            H[i+1, i] = 0.0
             
            rnrm = abs(s[i+1])
            if callback is not None:    
                callback(rnrm)
             
            if rnrm < tolb:
                y = solve(H[:i, :i],  s[:i])
                for k in range(0, i):
                    X += y[k]*V[k]
                info = 0
                return X, info

        y = solve(H[:m, :m],  s[:m])
        for k in range(0, k):
            X += y[k]*V[k]
        R = M1.solve(B - A.dot(X))
        rnrm = norm(R)
        if callback is not None:    
            callback(rnrm)
        if rnrm < tolb:
            info = 0
            break
    return X, info
