#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:18:45 2019

@author: bene
"""

import numpy as np
import scipy as sp
from  scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import bisect
from scipy import interpolate

def bspleval(x, knots, coeffs, order, debug=False):
    '''
    Found at https://stackoverflow.com/questions/22488637/getting-spline-equation-from-univariatespline-object
    Evaluate a B-spline at a set of points.

    Parameters
    ----------
    x : list or ndarray
        The set of points at which to evaluate the spline.
    knots : list or ndarray
        The set of knots used to define the spline.
    coeffs : list of ndarray
        The set of spline coefficients.
    order : int
        The order of the spline.

    Returns
    -------
    y : ndarray
        The value of the spline at each point in x.
    '''

    k = order
    t = knots
    m = np.alen(t)
    npts = np.alen(x)
    B = np.zeros((m-1,k+1,npts))

    if debug:
        print('k=%i, m=%i, npts=%i' % (k, m, npts))
        print('t=', t)
        print('coeffs=', coeffs)

    ## Create the zero-order B-spline basis functions.
    for i in range(m-1):
        B[i,0,:] = np.float64(np.logical_and(x >= t[i], x < t[i+1]))

    if (k == 0):
        B[m-2,0,-1] = 1.0

    ## Next iteratively define the higher-order basis functions, working from lower order to higher.
    for j in range(1,k+1):
        for i in range(m-j-1):
            if (t[i+j] - t[i] == 0.0):
                first_term = 0.0
            else:
                first_term = ((x - t[i]) / (t[i+j] - t[i])) * B[i,j-1,:]

            if (t[i+j+1] - t[i+1] == 0.0):
                second_term = 0.0
            else:
                second_term = ((t[i+j+1] - x) / (t[i+j+1] - t[i+1])) * B[i+1,j-1,:]

            B[i,j,:] = first_term + second_term
        B[m-j-2,j,-1] = 1.0

    if debug:
        plt.figure()
        for i in range(m-1):
            plt.plot(x, B[i,k,:])
        plt.title('B-spline basis functions')

    ## Evaluate the spline by multiplying the coefficients with the highest-order basis functions.
    y = np.zeros(npts)
    for i in range(m-k-1):
        y += coeffs[i] * B[i,k,:]

    if debug:
        plt.figure()
        plt.plot(x, y)
        plt.title('spline curve')
        plt.show()

    return(y)
    
x = np.array([-273.0, -176.4, -79.8, 16.9, 113.5, 210.1, 306.8, 403.4, 500.0])
y = np.array([2.25927498e-53, 2.56028619e-03, 8.64512988e-01, 6.27456769e+00, 1.73894734e+01,
        3.29052124e+01, 5.14612316e+01, 7.20531200e+01, 9.40718450e+01])

x_nodes = np.array([-273.0, -263.5, -234.8, -187.1, -120.3, -34.4, 70.6, 194.6, 337.8, 500.0])
y_nodes = np.array([2.25927498e-53, 3.83520726e-46, 8.46685318e-11, 6.10568083e-04, 1.82380809e-01,
                2.66344008e+00, 1.18164677e+01, 3.01811501e+01, 5.78812583e+01, 9.40718450e+01])

## Now get scipy's spline fit.
k = 3
tck = interpolate.splrep(x_nodes, y_nodes, k=k, s=0)
knots = tck[0]
coeffs = tck[1]
print('knot points=', knots)
print('coefficients=', coeffs)

## Now try scipy's object-oriented version. The result is exactly the same as "tck": the knots are the
## same and the coeffs are the same, they are just queried in a different way.
uspline = UnivariateSpline(x_nodes, y_nodes, s=0)
uspline_knots = uspline.get_knots()
uspline_coeffs = uspline.get_coeffs()

## Here are scipy's native spline evaluation methods. Again, "ytck" and "y_uspline" are exactly equal.
ytck = interpolate.splev(x, tck)
y_uspline = uspline(x)
y_knots = uspline(knots)

## Now let's try our manually-calculated evaluation function.
y_eval = bspleval(x, knots, coeffs, k, debug=False)

plt.plot(x, ytck, label='tck')
plt.plot(x, y_uspline, label='uspline')
plt.plot(x, y_eval, label='manual')

## Next plot the knots and nodes.
plt.plot(x_nodes, y_nodes, 'ko', markersize=7, label='input nodes')            ## nodes
plt.plot(knots, y_knots, 'mo', markersize=5, label='tck knots')    ## knots
plt.xlim((-300.0,530.0))
plt.legend(loc='best', prop={'size':14})

plt.figure()
plt.title('difference')
plt.plot(x, ytck-y_uspline, label='tck-uspl')
plt.plot(x, ytck-y_eval, label='tck-manual')
plt.legend(loc='best', prop={'size':14})

plt.show()