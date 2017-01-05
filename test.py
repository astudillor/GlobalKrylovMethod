#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from GlobalKrylovMethod import gmres
class convergeceHistory:
    def __init__(self):
        self.resvec = []
    def callback(self, _rnrm_):
        self.resvec.append(_rnrm_)
f = 'fs_680_1.mtx'
A = io.mmread(f).tocsr()
n = A.shape[0]
Xsol = np.ones(n)/float(n)
B = A.dot(Xsol)
bnrm = np.linalg.norm(B)
res = convergeceHistory()
X, info = gmres(A, B, tol = 1e-5,  = 200, callback = res.callback)
msg = "Info = {:d}, residual = {:f}, error = {:f}".format(info, 
       np.linalg.norm(B - A.dot(X))/bnrm, np.linalg.norm(Xsol - X))
print(msg)
plt.semilogy(res.resvec/bnrm)
plt.show()
