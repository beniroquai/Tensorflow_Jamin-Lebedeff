import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py 

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



'''CODE STARTS HERE'''
# Here we try to do some kind of phase unwrapping for quantitative Phase images coming from Jamin Lebedeff Microscope 
mymatfile = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/Jamin.Lebedeff/MATLAB/JL-FWD/JL_tensorflow.mat'

matname = 'myParameter'

#load system data; new MATLAB v7.3 Format! 
mat_matlab_data = h5py.File(mymatfile, 'r')
B_map = np.array(mat_matlab_data['B_mat'])
R_map = np.array(mat_matlab_data['R_mat'])
G_map = np.array(mat_matlab_data['G_mat'])
OPD_map = np.array(mat_matlab_data['OPD_mat'])

I_ref = np.array(mat_matlab_data['I_ref_mat'])
I_ref_R = np.squeeze(I_ref[0,:,:])
I_ref_G = np.squeeze(I_ref[1,:,:])
I_ref_B = np.squeeze(I_ref[2,:,:])
mysize = I_ref.shape

# Basically we want to get the minimum for TV(sqrt((R(OPD)-R'(OPD))**2-(G(OPD)-G'(OPD))**2-(B(OPD)-B'(OPD))**2))
# Tensorflow doesnt allow us to formulate this kind of slicing - ?! 
TF_opd = tf.Variable(1, tf.int32)
TF_B_map= tf.constant(B_map, tf.float32)
TF_R_map = tf.constant(R_map, tf.float32)
TF_G_map = tf.constant(G_map, tf.float32)
TF_OPD_map = tf.constant(OPD_map, tf.float32)

# Convert Image to Vector
TF_ref_R = tf.constant(I_ref_R)
TF_ref_R_vec = tf.reshape(TF_ref_R, [TF_ref_R.shape[0]*TF_ref_R.shape[1], 1])
TF_ref_G = tf.constant(I_ref_G)
TF_ref_G_vec = tf.reshape(TF_ref_G, [TF_ref_R.shape[0]*TF_ref_R.shape[1], 1])
TF_ref_B = tf.constant(I_ref_B)
TF_ref_B_vec = tf.reshape(TF_ref_B, [TF_ref_R.shape[0]*TF_ref_R.shape[1], 1])

# start Session 
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# better initialize with matlab opd result
batch_size = 1
Npixel = I_ref_R.shape[0]*I_ref_R.shape[1]
Nopd = OPD_map.shape[0]
opd_index = tf.one_hot(tf.random_uniform([batch_size, Npixel], minval=0, maxval=Nopd, dtype=tf.int32),Nopd)
      
# make sure dimensions are correct
TF_R_map = tf.expand_dims(tf.transpose(TF_R_map),0)
TF_G_map = tf.expand_dims(tf.transpose(TF_G_map),0)
TF_B_map = tf.expand_dims(tf.transpose(TF_B_map),0)

# stack the weighted color-vectors 
# The GUESS holds the current RGB values indexed by minimum L2 distance between RGB values in Experimental and LUT
# 1.) GUESS
myRGB_guess = []; 
myRGB_guess.append((TF_R_map*opd_index)); 
myRGB_guess.append(TF_G_map*opd_index); 
myRGB_guess.append(TF_B_map*opd_index)
myRGB_guess = tf.squeeze(tf.reduce_sum(tf.stack(myRGB_guess, 1),3))

# 2.) EXPERIMENTAL
myRGB_experimental = []; 
myRGB_experimental.append((TF_ref_R_vec)); 
myRGB_experimental.append(TF_ref_G_vec); 
myRGB_experimental.append(TF_ref_B_vec)
myRGB_experimental = tf.squeeze(tf.stack(myRGB_experimental, 0))

#%% formulate cost-fct 1:
# This one should reduce the L2 distance between the RGB Pixels to the one in the 
# RGB-OPD lookup-table
mySqrError = tf.reduce_mean(tf.pow(myRGB_guess-tf.cast(myRGB_experimental, tf.float32), 2))

#%% formulate cost-fct 2:
# we want to add a smootheness constraint on the result coming from L2 minimization, 
# This is done by adding TV-regularizer on the indexed image
TF_myopd = tf.reduce_sum(opd_index,2)
TF_myopd = tf.reshape(TF_myopd, ((mysize[1], mysize[1])))
myTVError = Reg_TV(TF_myopd)

lr = .1
# Define optimizers 
TF_opt_l2 = tf.train.AdamOptimizer(learning_Rate = lr)
TF_opt_l2.minimize(mySqrError)
TF_opt_TV = tf.train.AdamOptimizer(learning_Rate = lr)
TF_opt_TV.minimize(myTVError)


#%% start optimization part here
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(10):
    _, myopd = sess.run([TF_opt_l2, TF_myopd])
    _, myopd = sess.run([TF_opt_l2, TF_myopd])
    plt.imshow(myopd)
    