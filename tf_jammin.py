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


def Reg_NegSqr(toRegularize):
    mySqrt = tf.where( # Just affects the real part
                    tf.less(toRegularize , tf.zeros_like(toRegularize)),
                    tf_abssqr(toRegularize), tf.zeros_like(toRegularize))
     
    myReg = tf.reduce_mean(mySqrt)
    return myReg
 
def Reg_PosSqr(toRegularize):
    mySqrt = tf.where( # Just affects the real part
                    tf.greater(toRegularize , tf.zeros_like(toRegularize)),
                    tf_abssqr(toRegularize), tf.zeros_like(toRegularize))
     
    myReg = tf.reduce_mean(mySqrt)
    return myReg

def polyeval(x,coeff):
    # standard polynomial function for N-Th grade polynomial
    y = 0
    npoly = len(coeff)-1
    for i in np.linspace(npoly, 0, npoly+1):
        mycoeff = coeff[np.int32(i)]
        # print(str(mycoeff) + '*x^' + str(npoly-i))
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

def findOPD(RGBImg,R,G,B,OPDMax):
    # Minimal Norm solution for the OPD recovery from a look-up table

    # flatten arrays 
    R = np.expand_dims(np.expand_dims(R,0),0)
    G = np.expand_dims(np.expand_dims(G,0),0)
    B = np.expand_dims(np.expand_dims(B,0),0)    

    # Measure minimum L2 distance to closest Colorvalue
    R_val = np.repeat(np.expand_dims(RGBImg[0,:,:],-1), R.shape[-1], 2)
    G_val = np.repeat(np.expand_dims(RGBImg[1,:,:],-1), R.shape[-1], 2)
    B_val = np.repeat(np.expand_dims(RGBImg[2,:,:],-1), R.shape[-1], 2)
    
    myErr = (R_val - R)**2 + (G_val - G)**2 + (B_val - B)**2     

    OPDMap = np.argmin(myErr, axis=-1);
    return np.float32(OPDMap*OPDMax)
        
# https://github.com/tensorflow/tensorflow/issues/18383

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