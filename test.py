#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from GlobalKrylovMethod import gmres
n = 10
A = np.random.rand(n, n)
Xsol = np.random.rand(n, 1)
B = A.dot(Xsol)
X, info = gmres(A, B, m=n)
