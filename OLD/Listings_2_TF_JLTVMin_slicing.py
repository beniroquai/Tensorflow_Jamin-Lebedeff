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
B_map = (np.array(mat_matlab_data['B_mat']))
R_map = (np.array(mat_matlab_data['R_mat']))
G_map = (np.array(mat_matlab_data['G_mat']))
OPD_map = np.array(mat_matlab_data['OPD_mat'])

# flatten arrays 
I_exp = np.array(mat_matlab_data['I_ref_mat'])
R_exp = np.reshape(np.squeeze(I_exp[0,:,:]), (I_exp.shape[1]*I_exp.shape[2]))
G_exp = np.reshape(np.squeeze(I_exp[1,:,:]), (I_exp.shape[1]*I_exp.shape[2]))
B_exp = np.reshape(np.squeeze(I_exp[2,:,:]), (I_exp.shape[1]*I_exp.shape[2]))
mysize = I_exp.shape

# Convert Image to Vector
TF_R_exp = tf.constant(R_exp)
TF_G_exp = tf.constant(G_exp)
TF_B_exp = tf.constant(B_exp)

# Basically we want to get the minimum for TV(sqrt((R(OPD)-R'(OPD))**2-(G(OPD)-G'(OPD))**2-(B(OPD)-B'(OPD))**2))
# Tensorflow doesnt allow us to formulate this kind of slicing - ?! 
TF_opd = (tf.cast(tf.Variable(np.zeros((mysize[1]*mysize[2]))), tf.int32))
TF_R_map = tf.constant(np.squeeze(R_map)) # tf.constant((np.ones((mysize[1])*mysize[2])*R_map))
TF_G_map = tf.constant(np.squeeze(G_map))
TF_B_map = tf.constant(np.squeeze(B_map))

TF_OPD_map = tf.constant(OPD_map, tf.float32)

# get the RGB value according to a given OPD-index
TF_R_guess = tf.gather(TF_R_map, TF_opd, axis=-1)
TF_G_guess = tf.gather(TF_G_map, TF_opd, axis=-1)
TF_B_guess = tf.gather(TF_B_map, TF_opd, axis=-1)
 

# better initialize with matlab opd result

#%% formulate cost-fct 1:
# This one should reduce the L2 distance between the RGB Pixels to the one in the 
# RGB-OPD lookup-table
mySqrError = tf.reduce_mean((TF_R_guess-TF_R_exp)**2 + 
                            (TF_G_guess-TF_G_exp)**2 + 
                            (TF_B_guess-TF_B_exp)**2)

#%% formulate cost-fct 2:
# we want to add a smootheness constraint on the result coming from L2 minimization, 
# This is done by adding TV-regularizer on the indexed image
TF_myopd = tf.reshape(TF_opd, ((mysize[1], mysize[1])))
myTVError = Reg_TV(TF_myopd)

lr = .1
# Define optimizers 
TF_opt_l2 = tf.train.AdamOptimizer(learning_rate = lr)
TF_opt_l2.minimize(mySqrError)
TF_opt_TV = tf.train.AdamOptimizer(learning_rate = lr)
TF_opt_TV.minimize(myTVError)


#%% start optimization part here
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(10):
    _, myopd = sess.run([TF_opt_l2, TF_myopd])
    _, myopd = sess.run([TF_opt_l2, TF_myopd])
    plt.imshow(myopd)
    
    
    
