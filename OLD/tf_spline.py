#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 22:29:07 2019

@author: bene
"""

#!/usr/bin/env python
"""
    !!! Not certified fit for any purpose, use at your own risk !!!
    Copyright (c) Rex Sutton 2004-2017.
    Demo cubic spline fitting using tensor flow.
    Beware plays fast and loose with dimension checks.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def polyeval(x,coeff):
    # standard polynomial function for N-Th grade polynomial
    y = 0
    npoly = len(coeff)-1
    for i in np.linspace(npoly, 0, npoly+1):
        mycoeff = coeff[np.int32(i)]
        #print(str(mycoeff) + '*x^' + str(npoly-i))
        y += mycoeff*(x**(npoly-i))
    return y


def tf_abssqr(input):
    return tf.real(input*tf.conj(input))

def Reg_TV(toRegularize, BetaVals = [1,1], epsR = 1, epsC=1e-10, is_circ = True):
    # used rainers version to realize the tv regularizer   
    #% The Regularisation modification with epsR was introduced, according to
    #% Ferreol Soulez et al. "Blind deconvolution of 3D data in wide field fluorescence microscopy
    #%
    #
    #function [myReg,myRegGrad]=RegularizeTV(toRegularize,BetaVals,epsR)
    #epsC=1e-10;

    
    if(is_circ):
        aGradL_1 = (toRegularize - tf.manip.roll(toRegularize, 1, 0))/BetaVals[0]
        aGradL_2 = (toRegularize - tf.manip.roll(toRegularize, 1, 1))/BetaVals[1]

        aGradR_1 = (toRegularize - tf.manip.roll(toRegularize, -1, 0))/BetaVals[0]
        aGradR_2 = (toRegularize - tf.manip.roll(toRegularize, -1, 1))/BetaVals[1]
        
        print('We use circular shift for the TV regularizer')
    else:    
        toRegularize_sub = toRegularize[1:-2,1:-2,1:-2]
        aGradL_1 = (toRegularize_sub - toRegularize[2:-1,1:-2,1:-2])/BetaVals[0] # cyclic rotation
        aGradL_2 = (toRegularize_sub - toRegularize[1:-1-1,2:-1,1:-1-1])/BetaVals[1] # cyclic rotation
        
        aGradR_1 = (toRegularize_sub - toRegularize[0:-3,1:-2,1:-2])/BetaVals[0] # cyclic rotation
        aGradR_2 = (toRegularize_sub - toRegularize[1:-2,0:-3,1:-2])/BetaVals[1] # cyclic rotation
            
    mySqrtL = tf.sqrt(tf_abssqr(aGradL_1)+tf_abssqr(aGradL_2)+epsR)
    mySqrtR = tf.sqrt(tf_abssqr(aGradR_1)+tf_abssqr(aGradR_2)+epsR)
     
    mySqrt = mySqrtL + mySqrtR; 
    
    if(1):
        mySqrt = tf.where(
                    tf.less(mySqrt , epsC*tf.ones_like(mySqrt)),
                    epsC*tf.ones_like(mySqrt),
                    mySqrt) # To avoid divisions by zero
    else:               
        mySqrt = mySqrt # tf.clip_by_value(mySqrt, 0, np.inf)    
        

        
    myReg = tf.reduce_mean(mySqrt)

    return myReg



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



def bspleval_tf(x, knots, coeffs, order, debug=False):
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

    with tf.name_scope('initialize'):
        k = order
        t = knots
        m = np.alen(t)
        npts = np.alen(x)
        shape = (m-1,k+1,npts)
        B = tf.zeros(shape)

    if debug:
        print('k=%i, m=%i, npts=%i' % (k, m, npts))
        print('t=', t)
        print('coeffs=', coeffs)

    ## Create the zero-order B-spline basis functions.
    # hack-around - TF doesn't allow tensor-assignement    
    B_tmp = []
    with tf.name_scope('create_B_matrix'):
        for i in range(m-1):
            data = (tf.cast(tf.math.logical_and(tf.greater_equal(x, t[i]),(tf.less_equal(x,t[i+1]))), tf.float64))
            B_tmp.append(data)
        B_tmp = tf.stack(B_tmp)
        B = []
        for i in range(k+1):
            B.append(B_tmp)
        B = tf.Variable(tf.cast(tf.transpose(tf.stack(B), [1,0,2]), tf.float32))
        
        if (k == 0):
            B[m-2,0,-1] = 0*1.0
        

    ## Next iteratively define the higher-order basis functions, working from lower order to higher.
    for jj in range(1,k+1):
        with tf.name_scope('jj_'+str(jj)):
            for i in range(m-jj-1):
                with tf.name_scope('i_'+str(i)):
    
                    if (t[i+jj] - t[i] == 0.0):
                        first_term = 0.0
                    else:
                        first_term = tf.cast(((x - t[i]) / (t[i+jj] - t[i])), tf.float32) * B[i,jj-1,:]
        
                    if (t[i+jj+1] - t[i+1] == 0.0):
                        second_term = 0.0
                    else:
                        second_term = tf.cast(((t[i+jj+1] - x) / (t[i+jj+1] - t[i+1])), tf.float32) * B[i+1,jj-1,:]
        
                    data = first_term + second_term
                    
                    #B[i,j,:] = first_term + second_term - https://github.com/tensorflow/tensorflow/issues/18383
                    #indices = [[m-jj-2],[jj],int(np.linspace(0,49))]
                    #update = np.ones(49)*data
                    #indices = tf.squeeze(tf.where(tf.greater(tf.abs(update), -1)))
                    #B = tf.scatter_update(tf.Variable(B), indices, update)

                    B = replace_slice_in(tf.cast(B, tf.float32))[i,jj,:].with_value([tf.cast(data, tf.float32)])
    
            #B[m-j-2,j,-1] = 1.0        
            B = replace_slice_in(tf.cast(B, tf.float32))[m-jj-2,jj,1].with_value([tf.cast(1., tf.float32)])
            #indices = [[m-jj-2],[jj],[49]]
            #print(indices)
            #B = tf.scatter_update((B), indices, 1.0)

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
    
    
def myPolyFunc(x, myPar):
    # Simple polynomial function to fit the data to
    y = 0
    try:
        npoly = myPar.get_shape().as_list()[0] # Tensorflow
    except:
        npoly = myPar.shape[0] # numpy

    if(0):
        for myiter in range(npoly):
            y += myPar[myiter]*x**(myiter)# + myPar[1]*x +  myPar[2]*x**2 + myPar[3] * myPar[4]**3 + myPar[5]**5 
    else:
        try:
            y = myPar[0] + np.sin(x*myPar[1]+myPar[2])*myPar[3] +  myPar[4]*x + np.exp(x*myPar[5]+myPar[6]) + myPar[7]*x**2 + myPar[8]*myPar[9]**3+myPar[10]**4+myPar[11]**5+myPar[12]**6+myPar[13]**7+myPar[14]**8 # numpy
        except:
            y = myPar[0] + tf.sin(x*myPar[1]+myPar[2])*myPar[3] +  myPar[4]*x + tf.exp(x*myPar[5]+myPar[6]) + myPar[7]*x**2 + myPar[8]*myPar[9]**3+myPar[10]**4+myPar[11]**5+myPar[12]**6+myPar[13]**7+myPar[14]**8 # Tensorflow
            
    
    return y


def tridiag(lower, diag, upper):
    """
    Make matrix from tri-diagonal representation.
    Args:
        lower: Lower diagonal.
        diag: The diagonal.
        upper: The upper diagonal.
    Returns:
        Tri-diagonal matrix.
    """
    return np.diag(lower, -1) + np.diag(diag, 0) + np.diag(upper, 1)


def cubic_spline_xdeps(x_axis):
    """
    Calculate the dependencies on the x-axis as far as we can go on the CPU.
    Args:
        x_axis: The independent variable.
    Returns:
        A tuple, The A matrix, the differences between consecutive axis values, three constants.
    """
    # pylint: disable=invalid-name

    dx = np.diff(x_axis)
    n = x_axis.shape[0]

    A = np.zeros((3, n), x_axis.dtype)  # This is a banded matrix representation.

    A[0, 2:] = dx[:-1]                   # The upper diagonal
    A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])  # The diagonal
    A[2, :-2] = dx[1:]                   # The lower diagonal

    A[1, 0] = dx[1]
    A[0, 1] = x_axis[2] - x_axis[0]

    A[1, -1] = dx[-2]
    A[2, -2] = x_axis[-1] - x_axis[-3]

    A_upper = A[0, 1:]
    A_diag = A[1,]
    A_lower = A[2, :-1]

    A = tridiag(A_lower, A_diag, A_upper)
    return A, dx, x_axis[2] - x_axis[0], x_axis[-1] - x_axis[-3]
    # pylint: enable=invalid-name


def cubic_spline_coefficients(x, y_tensor): # pylint: disable=too-many-locals
    """
    Compute the cubic spline co-efficients given the dependent variable and
        the independent variable.
     NOTE probably fails for len(x) <4 .
    Args:
        x: The independent variable in numpy array.
        y_tensor: The dependent variable .
    Returns:
        Cubic spline co-efficients in tensors.
    """
    # pylint: disable=invalid-name
    n = x.shape[0]

    A, dx, d1, d2 = cubic_spline_xdeps(x)

    slope_tensor = y_tensor[1:] - y_tensor[:-1]

    b_beg_tensor = ((dx[0] + 2*d1) * dx[1] * slope_tensor[0:1] + dx[0]**2 * slope_tensor[1:2]) / d1
    b_mid_tensor = 3.0 * (dx[1:] * slope_tensor[:-1] + dx[:-1] * slope_tensor[1:])
    b_end_tensor = ((dx[-1]**2*slope_tensor[n-3:n-2]
                     + (2*d2 + dx[-1])*dx[-2]*slope_tensor[n-2:n-1]) / d2)

    B_tensor = tf.concat([b_beg_tensor, tf.concat([b_mid_tensor, b_end_tensor], 0)], 0)
    B_tensor = tf.reshape(B_tensor, [n, 1])

    A_tensor = tf.constant(A)
    # No tridiag solve in Tensorflow.
    s_tensor = tf.matrix_solve(A_tensor, B_tensor)
    flat_s_tensor = tf.reshape(s_tensor, [n])

    t_tensor = (flat_s_tensor[:-1] + flat_s_tensor[1:] - 2.0 * slope_tensor) / dx

    c0_tensor = t_tensor / dx
    c1_tensor = (slope_tensor - flat_s_tensor[:-1]) / dx - t_tensor
    c2_tensor = flat_s_tensor[:-1]
    c3_tensor = y_tensor[:-1]

    return c0_tensor, c1_tensor, c2_tensor, c3_tensor
    # pylint: enable=invalid-name


def tile_columns(vec, num_cols):
    """
    Build a matrix tensor of num_cols copies of a vector.
    Args:
        vec: The vector.
        num_cols: The number of columns.
    Returns:
        A matrix tensor of num_cols copies of a vector.
    """
    num_rows = vec.get_shape()[0]
    temp1 = tf.tile(vec, [num_cols])
    temp2 = tf.reshape(temp1, np.array([num_cols, num_rows]))
    return tf.transpose(temp2)


def compute_range_selection(axis, x): # pylint: disable=too-many-locals
    """
    Args:
        axis: A strictly increasing vector of floats, describing intervals.
        x: The items that are binned into the intervals.
    Returns:
        A matrix, masking the items to be be selected.
    """
    # pylint: disable=invalid-name
    axis_lower = axis[:-1]
    axis_upper = axis[1:]

    n = len(axis_lower)
    nx = int(x.get_shape()[0])

    al = tf.constant(axis_lower)
    alx = tile_columns(al, nx)

    au = tf.constant(axis_upper)
    aux = tile_columns(au, nx)

    xxx = tf.reshape(tf.tile(x, [n]), [n, nx])

    return tf.transpose(tf.logical_and(tf.less(xxx, aux), tf.greater_equal(xxx, alx)))
    # pylint: enable=invalid-name


def select(condition, items): # pylint: disable=too-many-locals
    """
    Args:
        condition: Bool matrix indicating the item intervals.
        items: The items to select from based on the interval, lenth len(axis)-1
    Returns:
        A vector of length x, of the items in q corresponding to the condition.
    """
    # pylint: disable=invalid-name
    num_rows = int(condition.get_shape()[0])
    num_cols = int(condition.get_shape()[1])
    return tf.boolean_mask(tf.reshape(tf.tile(items, [num_rows]), [num_rows, num_cols]), condition)
    # pylint: enable=invalid-name


def cubic(c0, c1, c2, c3, x): # pylint: disable=invalid-name
    """
        Evaluate the cubic polynomial given by co-efficients.
    Args:
        c0: First co-efficient.
        c1: Second co-efficient.
        c2: Third co-efficient.
        c3: Fourth co-efficient.
        x: The argument to the spline.
    Returns:
        The value of the cubic polynomial.
    """
    # pylint: disable=invalid-name
    xx = x*x
    xxx = x*xx
    return c0 * xxx + c1 * xx + c2 * x + c3
    # pylint: enable=invalid-name


def cubic_spline(independent_variable, dependent_variable, points, np_dtype=np.float32):
    """
    Args:
        independent_variable: The independent variable.
        dependent_variable: The dependent variable.
        points: The independent variables at which to evaluate the spline.
        np_dtype: The dtype.
    Returns:
        A tensor containing the values of the cubic spline at the points.
    """

    if independent_variable.dtype != np_dtype:
        raise ValueError("Axis(independent_variable)"
                         " and values(dependent_variable) must be compatible dtypes.")

    if points.dtype != dependent_variable.dtype:
        raise ValueError("Values and interpolation points must be compatible dtypes.")

    coefficients = cubic_spline_coefficients(independent_variable, dependent_variable)
    mask = compute_range_selection(independent_variable, points)
    c0_coeffs_tensor = select(mask, coefficients[0])
    c1_coeffs_tensor = select(mask, coefficients[1])
    c2_coeffs_tensor = select(mask, coefficients[2])
    c3_coeffs_tensor = select(mask, coefficients[3])

    lower_bounds = tf.constant(independent_variable[:-1])
    deltas = points - select(mask, lower_bounds)

    return cubic(c0_coeffs_tensor,
                 c1_coeffs_tensor,
                 c2_coeffs_tensor,
                 c3_coeffs_tensor,
                 deltas)


def main():
    """
    Plot an example fitting through sinusoidal, with 80,000 data points,
     and compute gradients of the interpolated points with respect to the dependent variable.
    """
    # pylint: disable=invalid-name
    x = np.arange(10.0)
    x = x.astype(np.float32, copy=False)
    y = np.sin(x)*x**2+x
    y = y.astype(np.float32, copy=False)
    xs = np.arange(0.5, 8.5, 0.0001)
    xs = xs.astype(np.float32, copy=False)
    
    y_tensor = tf.constant(y)
    xs_tensor = tf.constant(xs)
    ys_tensor = cubic_spline(x, y_tensor, xs_tensor)
    
    gradients_tensor = tf.gradients(ys_tensor, y_tensor)
    
    
    session = tf.Session()
    res = session.run([ys_tensor])#, gradients_tensor])
    ys = res[0]
    
    print("Num data points:", len(xs)) # 80000 data points
    print("Gradients:")
    #print(res[1])
    
    plt.figure(figsize=(6.5, 4))
    plt.plot(x, y, label='data')
    plt.plot(xs, ys, label='spline')
    
    plt.xlim(-0.5, 9.5)
    plt.legend(loc='lower right', ncol=2)
    plt.show()
    # pylint: enable=invalid-name

main()    
    
# https://github.com/tensorflow/tensorflow/issues/18383
import tensorflow as tf

def replace_slice(input_, replacement, begin, size=None):
    inp_shape = tf.shape(input_)
    if size is None:
        size = tf.shape(replacement)
    else:
        replacement = tf.broadcast_to(replacement, size)
    padding = tf.stack([begin, inp_shape - (begin + size)], axis=1)
    replacement_pad = tf.pad(replacement, padding)
    mask = tf.pad(tf.ones_like(replacement, dtype=tf.bool), padding)
    return tf.where(mask, replacement_pad, input_)

def replace_slice_in(tensor):
    return _SliceReplacer(tensor)

class _SliceReplacer:
    def __init__(self, tensor):
        self._tensor = tensor
    def __getitem__(self, slices):
        return _SliceReplacer._Inner(self._tensor, slices)
    def with_value(self, replacement):  # Just for convenience in case you skip the indexing
        return _SliceReplacer._Inner(self._tensor, (...,)).with_value(replacement)
    class _Inner:
        def __init__(self, tensor, slices):
            self._tensor = tensor
            self._slices = slices
        def with_value(self, replacement):
            begin, size = _make_slices_begin_size(self._tensor, self._slices)
            return replace_slice(self._tensor, replacement, begin, size)

# This computes begin and size values for a set of slices
def _make_slices_begin_size(input_, slices):
    if not isinstance(slices, (tuple, list)):
        slices = (slices,)
    inp_rank = tf.rank(input_)
    inp_shape = tf.shape(input_)
    # Did we see a ellipsis already?
    before_ellipsis = True
    # Sliced dimensions
    dim_idx = []
    # Slice start points
    begins = []
    # Slice sizes
    sizes = []
    for i, s in enumerate(slices):
        if s is Ellipsis:
            if not before_ellipsis:
                raise ValueError('Cannot use more than one ellipsis in slice spec.')
            before_ellipsis = False
            continue
        if isinstance(s, slice):
            start = s.start
            stop = s.stop
            if s.step is not None:
                raise ValueError('Step value not supported.')
        else:  # Assumed to be a single integer value
            start = s
            stop = s + 1
        # Dimension this slice refers to
        i_dim = i if before_ellipsis else inp_rank - (len(slices) - i)
        dim_size = inp_shape[i_dim]
        # Default slice values
        start = start if start is not None else 0
        stop = stop if stop is not None else dim_size
        # Fix negative indices
        start = tf.cond(tf.convert_to_tensor(start >= 0), lambda: start, lambda: start + dim_size)
        stop = tf.cond(tf.convert_to_tensor(stop >= 0), lambda: stop, lambda: stop + dim_size)
        dim_idx.append([i_dim])
        begins.append(start)
        sizes.append(stop - start)
    # For empty slice specs like [...]
    if not dim_idx:
        return tf.zeros_like(inp_shape), inp_shape
    # Make full begin and size array (including omitted dimensions)
    begin_full = tf.scatter_nd(dim_idx, begins, [inp_rank])
    size_mask = tf.scatter_nd(dim_idx, tf.ones_like(sizes, dtype=tf.bool), [inp_rank])
    size_full = tf.where(size_mask,
                         tf.scatter_nd(dim_idx, sizes, [inp_rank]),
                         inp_shape)
    return begin_full, size_full