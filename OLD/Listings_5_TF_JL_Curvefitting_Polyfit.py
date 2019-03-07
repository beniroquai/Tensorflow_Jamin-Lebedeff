import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py 
import tf_jammin as tf_jammin
from mpl_toolkits.mplot3d.axes3d import Axes3D
#%load_ext autoreload
#%autoreload 2
mpl.rc('figure',  figsize=(10, 5))
#mpl.rc('image', cmap='gray')





''' Abstract ''' 
# This code is reconstructing the optical path difference (OPD) from RGB images
# acquired with a cellphone camera. This is done by providing an RGB-calibration 
# measurement with known OPD (e.g. an optical fiber). Currently this OPD-map 
# is computed in MATLAB and preloaded as a .mat file. 
# The idea is to minimize the L2 distance between a measured RGB-pixel and the
# previously generated OPD/RGB-lookup table. In order to minimize the effect of 
# phase discontuinites, we add TV regularization and use standard Gradient-Descent
# to minimize a combined loss-function:
# min -> OPD ||(R(OPD)-R_meas)^2 +  (G(OPD)-G_meas)^2 +  (B(OPD)-B_meas)^2 +|| + TV(OPD)

''' Variable Declaration '''
mymatfile = './JL_tensorflow.mat' # Exported MAT-file, contains OPD/R/G/B-Maps

# Fitting-related
n_poly = 11 # order of polynomial fitted to the data
opdmax = 1650 # maximum optical path-difference in [nm]

# Optmization-related 
lambda_neg = 100 # Negative/Positive penalty
lambda_tv = 1000 # TV-parameter
epsC = 1e0 # TV-parameter
lr = 100 # learning-rate
Niter = 50 # number of iteration
is_debug = False # Show results while iterating? 


''' Preload MATLAB Data '''
# load system data; new MATLAB v7.3 Format! 
mat_matlab_data = h5py.File(mymatfile, 'r')
OPD_mask = np.squeeze(np.array(mat_matlab_data['mask_mat']))
myopd_res_matlab = np.squeeze(np.array(mat_matlab_data['OPDMap_mat']))
B_map = np.squeeze(np.array(mat_matlab_data['B_mat']))
R_map = np.squeeze(np.array(mat_matlab_data['R_mat']))
G_map = np.squeeze(np.array(mat_matlab_data['G_mat']))
OPD_map = np.squeeze(np.array(mat_matlab_data['OPD_mat']))
I_exp = np.array(mat_matlab_data['I_ref_mat']) # Value we want to reconstruct
nopdsteps = OPD_map.shape[0] # Number of the quantized OPD-RGB look-up values
mulfac = opdmax/nopdsteps # scale the OPD according to the matlab values 
myinitopd = mulfac*myopd_res_matlab 

# Display the curves 
plt.figure()
plt.title('Curves for RGB')
plt.plot(OPD_map,R_map, 'x')
plt.plot(OPD_map,B_map, 'x')
plt.plot(OPD_map,G_map, 'x')
plt.legend(['R', 'G','B'])
plt.show()


''' Fit the RGB data to a spline to remove possible noise '''    
# fit function using scipy
R_fit_func = np.poly1d(np.polyfit(np.squeeze(OPD_map), R_map, n_poly))
G_fit_func = np.poly1d(np.polyfit(np.squeeze(OPD_map), G_map, n_poly))
B_fit_func = np.poly1d(np.polyfit(np.squeeze(OPD_map), B_map, n_poly))

# eval fit
R_fit = tf_jammin.polyeval((OPD_map), np.squeeze(R_fit_func.coeffs))
G_fit = tf_jammin.polyeval((OPD_map), np.squeeze(G_fit_func.coeffs))
B_fit = tf_jammin.polyeval((OPD_map), np.squeeze(B_fit_func.coeffs))

# fit of R-curve
plt.subplot(131)
plt.title('R')
plt.plot(OPD_map, R_map, 'x')
plt.plot(OPD_map, R_fit, '-')
plt.legend('Numpy', 'Own')

# fit of G-curve
plt.subplot(132)
plt.title('G')
plt.plot(OPD_map, G_map, 'x')
plt.plot(OPD_map, G_fit, '-')
plt.legend('Numpy', 'Own')

# fit of G-curve
plt.subplot(133)
plt.title('B')
plt.plot(OPD_map, B_map, 'x')
plt.plot(OPD_map, B_fit, '-')
plt.legend('Numpy', 'Own')
plt.show()

# RGB-OPD parametric curve
fig8 = plt.figure()
ax8 = fig8.gca(projection = '3d')
ax8.plot(R_map, G_map, B_map)
ax8.plot(R_fit, G_fit, B_fit)
plt.show()

# Test the result with given minimal norm solution and fitted data
RGB_result_matlab = np.zeros((myopd_res_matlab.shape[0],myopd_res_matlab.shape[1],3))
RGB_result_matlab[:,:,0] = tf_jammin.polyeval(mulfac*myopd_res_matlab, np.squeeze(R_fit_func.coeffs))
RGB_result_matlab[:,:,1] = tf_jammin.polyeval(mulfac*myopd_res_matlab, np.squeeze(G_fit_func.coeffs))
RGB_result_matlab[:,:,2] = tf_jammin.polyeval(mulfac*myopd_res_matlab, np.squeeze(B_fit_func.coeffs))

# show the simulated result according to fitted RGB Data
plt.imshow(RGB_result_matlab/np.max(RGB_result_matlab)), plt.colorbar()
plt.imshow(RGB_result_matlab/np.max(RGB_result_matlab)), plt.colorbar()


''' TENSORFLOW STARTS HERE '''
# =============================================================================
# #%% Formulate the imaging model 
# =============================================================================
# Basically we want to get the minimum for 
# ||Ax-f|| +  TV(sqrt((R(OPD)-R'(OPD))**2-(G(OPD)-G'(OPD))**2-(B(OPD)-B'(OPD))**2))

# seperate calibration arrays into RGB
R_exp = np.squeeze(I_exp[0,:,:])
G_exp = np.squeeze(I_exp[1,:,:])
B_exp = np.squeeze(I_exp[2,:,:])
mysize = I_exp.shape

# Convert Image to Tensorflow objects 
TF_R_exp = tf.constant(R_exp)
TF_G_exp = tf.constant(G_exp)
TF_B_exp = tf.constant(B_exp)
TF_R_map = tf.constant(np.squeeze(R_map)) 
TF_G_map = tf.constant(np.squeeze(G_map))
TF_B_map = tf.constant(np.squeeze(B_map))
OPD_mask_flat = tf.Variable(np.reshape(OPD_mask, OPD_mask.shape[0]*OPD_mask.shape[1])) 

# This is the matlab reconstruction (Minimum Norm) #TODO: We want to compute this in Python too!
TF_opd = tf.Variable(myinitopd) 

# We only want to update the inner part of the mask (where OPD_mask is greater than 0)
updates = tf.boolean_mask(TF_opd, OPD_mask>0) # TF_opd*OPD_mask
indexes = tf.cast(tf.where(OPD_mask > 0), tf.int32)
TF_opd_masked = tf.scatter_nd(indexes, updates , tf.shape(OPD_mask))

# Compute the "Guess" based on the Variable OPD
TF_R_guess = tf_jammin.polyeval(TF_opd_masked, np.squeeze(R_fit_func.coeffs))
TF_G_guess = tf_jammin.polyeval(TF_opd_masked, np.squeeze(G_fit_func.coeffs))
TF_B_guess = tf_jammin.polyeval(TF_opd_masked, np.squeeze(B_fit_func.coeffs))

''' formulate cost-fct 1:'''
# we want to add a smootheness constraint on the result coming from L2 minimization, 
# This is done by adding TV-regularizer on the indexed image
# This one should reduce the L2 distance between the RGB Pixels to the one in the 
# RGB-OPD lookup-table
TF_mySqrError = tf.reduce_mean(((TF_R_guess-TF_R_exp)**2 + 
                            (TF_G_guess-TF_G_exp)**2 + 
                            (TF_B_guess-TF_B_exp)**2))

# in order to have a smooth phase without discontinuities we want to have a small TV norm
TF_myTVError = lambda_tv * tf_jammin.Reg_TV(TF_opd_masked, epsC=epsC)

# we don't want to have negative values 
TF_neg_loss = lambda_neg * tf_jammin.Reg_NegSqr(TF_opd_masked) # clip values if out of range (no suppot)
TF_pos_loss = lambda_neg * tf_jammin.Reg_PosSqr(TF_opd_masked-np.max(OPD_map)) # clip values if out of range (no suppot)

# Placeholder for the learningrate 
TF_lr = tf.placeholder(tf.float32, shape=[])

# Combined loss
TF_myError = TF_mySqrError + TF_myTVError # this works best! 

''' Define optimizers '''
TF_opt_l2 = tf.train.AdamOptimizer(learning_rate = TF_lr)
TF_loss_l2 = TF_opt_l2.minimize(TF_mySqrError)
TF_opt_TV = tf.train.AdamOptimizer(learning_rate = TF_lr)
TF_loss_TV = TF_opt_TV.minimize(TF_myTVError)
TF_opt = tf.train.AdamOptimizer(learning_rate = TF_lr)
TF_loss = TF_opt_TV.minimize(TF_myError)

''' start optimization part here '''
sess = tf.Session()
sess.run(tf.global_variables_initializer())

myopd_old = sess.run(TF_opd)
plt.imshow(myopd_old), plt.colorbar(), plt.show()


for i in range(Niter):
    # Alternating? - Better not! 
    #my_loss_l2,_ = sess.run([TF_mySqrError, TF_loss_l2], feed_dict={TF_lr:lr})
    #my_loss_tv,_ = sess.run([TF_myTVError,TF_loss_TV], feed_dict={TF_lr:lr})

    # combined loss works best
    my_loss_tv,my_loss_l2,_ = sess.run([TF_myTVError,TF_mySqrError,TF_loss], feed_dict={TF_lr:lr}) 
    
    #my_loss_tv,my_loss_l2,_ = sess.run([TF_myTVError,TF_mySqrError,TF_loss], feed_dict={TF_lr:lr}) 
    print("My Loss L2: "+str(my_loss_l2)+", My Loss TV: "+str(my_loss_tv))
    
    if(is_debug):
        myopd_new = sess.run(TF_opd_masked)
        plt.imshow(myopd_new), plt.colorbar(), plt.show()

#%%
plt.subplot(131)
plt.imshow(myopd_old), plt.colorbar(), plt.title('Minimal Norm solution')
plt.subplot(132)
plt.imshow(myopd_new), plt.colorbar(), plt.title('Gradient Descent + TV')
plt.subplot(133)
plt.title('Crosssection through the center')
plt.plot(myopd_old[:,myopd_old.shape[1]//2])
plt.plot(myopd_new[:,myopd_old.shape[1]//2]), plt.legend('Old', 'New'), plt.show()
