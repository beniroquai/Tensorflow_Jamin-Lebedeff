import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py 
import tf_jammin as tf_jammin
import skvideo.io

import NanoImagingPack as nip
from mpl_toolkits.mplot3d.axes3d import Axes3D

import tifffile as tif
#%load_ext autoreload
#%autoreload 2
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')

nip.config.__DEFAULTS__['IMG_VIEWER']='NIP_VIEW'


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
mymatfile = './JL_tensorflow_19_5_20.mat'
myvideopath = './data/HUAWEI/'
myvideofile = 'VID_20190520_135445.mp4'


#%%
myroisize = 750 # a subregion of the image will be cropped around the center in order to reduce the computational time
use_mask = False # do you want to use a mas for restricting the recovery?

# Fitting-related
n_poly = 11 # order of polynomial fitted to the data
opdmax = 1650 # maximum optical path-difference in [nm]
use_matlab = False # USe values stored in matlab .MAT?

# Optmization-related 
lambda_tv = 50 # TV-parameter
epsC = 1e-2 # TV-parameter
#lambda_neg = 100 # Negative/Positive penalty
lr = 100 # learning-rate
Niter = 100 # number of iteration
is_debug = False # Show results while iterating? 

Ndisplay_text = 10
Ndisplay = 100


#%%
''' Read the videofile '''
myvideo = myvideopath+myvideofile
videogen = skvideo.io.vreader(myvideo)

myimage = []

iframe = 0
for frame in videogen:
    rgb_frame = frame
    myimage.append(np.uint8(rgb_frame))
    iframe += 1
    print(iframe)
    if iframe> 10000:
        break

# extract subroi    
myallimages = np.transpose(np.array(myimage),[1,2,3,0])
myimage_size = myallimages.shape[0:2]
myallimages_sub = nip.extract(myallimages, (myroisize, myroisize, 3, myallimages.shape[3]))

''' Preload MATLAB Data '''
# load system data; new MATLAB v7.3 Format! 
mat_matlab_data = h5py.File(mymatfile, 'r')
#OPD_mask = np.squeeze(np.array(mat_matlab_data['mask_mat']))
myopd_res_matlab = np.squeeze(np.array(mat_matlab_data['OPDMap_mat']))
B_map = np.squeeze(np.array(mat_matlab_data['B_mat']))
R_map = np.squeeze(np.array(mat_matlab_data['R_mat']))
G_map = np.squeeze(np.array(mat_matlab_data['G_mat']))
OPD_map = np.squeeze(np.array(mat_matlab_data['OPD_mat']))

nopdsteps = OPD_map.shape[0] # Number of the quantized OPD-RGB look-up values
mulfac = opdmax/nopdsteps # scale the OPD according to the matlab values 
OPD_mask = np.ones((myroisize,myroisize)) # we don't want a mask here


''' Fit the RGB data to a spline to remove possible noise '''    
# fit function using scipy
R_fit_func = np.poly1d(np.polyfit(np.squeeze(OPD_map), R_map, n_poly))
G_fit_func = np.poly1d(np.polyfit(np.squeeze(OPD_map), G_map, n_poly))
B_fit_func = np.poly1d(np.polyfit(np.squeeze(OPD_map), B_map, n_poly))

# eval fit
R_fit = tf_jammin.polyeval((OPD_map), np.squeeze(R_fit_func.coeffs))
G_fit = tf_jammin.polyeval((OPD_map), np.squeeze(G_fit_func.coeffs))
B_fit = tf_jammin.polyeval((OPD_map), np.squeeze(B_fit_func.coeffs))


# Test the result with given minimal norm solution and fitted data
RGB_result_matlab = np.zeros((myopd_res_matlab.shape[0],myopd_res_matlab.shape[1],3))
RGB_result_matlab[:,:,0] = tf_jammin.polyeval(mulfac*myopd_res_matlab, np.squeeze(R_fit_func.coeffs))
RGB_result_matlab[:,:,1] = tf_jammin.polyeval(mulfac*myopd_res_matlab, np.squeeze(G_fit_func.coeffs))
RGB_result_matlab[:,:,2] = tf_jammin.polyeval(mulfac*myopd_res_matlab, np.squeeze(B_fit_func.coeffs))

# show the simulated result according to fitted RGB Data
plt.imshow(RGB_result_matlab/np.max(RGB_result_matlab)), plt.colorbar(), plt.show()



''' TENSORFLOW STARTS HERE '''
# =============================================================================
# #%% Formulate the imaging model 
# =============================================================================
# Basically we want to get the minimum for 
# ||Ax-f|| +  TV(sqrt((R(OPD)-R'(OPD))**2-(G(OPD)-G'(OPD))**2-(B(OPD)-B'(OPD))**2))


# Convert Image to Tensorflow objects 
mysize = np.array((3,myroisize,myroisize))
TF_R_exp = tf.placeholder(dtype=tf.float32,shape=(myroisize,myroisize))
TF_G_exp = tf.placeholder(dtype=tf.float32,shape=(myroisize,myroisize))
TF_B_exp = tf.placeholder(dtype=tf.float32,shape=(myroisize,myroisize))
TF_R_map = tf.constant(np.squeeze(R_map)) 
TF_G_map = tf.constant(np.squeeze(G_map))
TF_B_map = tf.constant(np.squeeze(B_map))

# Placeholder for the learningrate 
TF_lr = tf.placeholder(tf.float32, shape=[])
TF_lambda_TV = tf.placeholder(tf.float32, shape=[])
TF_epsC = tf.placeholder(tf.float32, shape=[])

# This is the matlab reconstruction (Minimum Norm) #TODO: We want to compute this in Python too!
TF_opd = tf.Variable(np.zeros((myroisize,myroisize))) 

# We only want to update the inner part of the mask (where OPD_mask is greater than 0)
updates = tf.boolean_mask(TF_opd, OPD_mask>0) # TF_opd*OPD_mask
indexes = tf.cast(tf.where(OPD_mask > 0), tf.int32)
TF_opd_masked = tf.cast(tf.scatter_nd(indexes, updates , tf.shape(OPD_mask)), tf.float32)

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
TF_myTVError = TF_lambda_TV * tf_jammin.Reg_TV(TF_opd_masked, epsC=TF_epsC)


# we don't want to have negative values 
lambda_neg = 1000
TF_neg_loss = lambda_neg * tf_jammin.Reg_NegSqr(TF_opd_masked) # clip values if out of range (no suppot)
TF_pos_loss = lambda_neg * tf_jammin.Reg_PosSqr(TF_opd_masked-np.max(OPD_map)) # clip values if out of range (no suppot)

# Combined loss
TF_myError = TF_mySqrError + TF_myTVError # this works best! 

''' Define optimizers '''
TF_opt_l2 = tf.train.AdamOptimizer(learning_rate = TF_lr)
TF_loss_l2 = TF_opt_l2.minimize(TF_mySqrError)
TF_opt_TV = tf.train.AdamOptimizer(learning_rate = TF_lr)
TF_loss_TV = TF_opt_TV.minimize(TF_myTVError)
TF_opt = tf.train.AdamOptimizer(learning_rate = TF_lr)
TF_loss = TF_opt_TV.minimize(TF_myError)

#%%
''' start optimization part here '''
allframes_new = []
allframes_old = []
allframes_raw = []

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# iterate over all images
for iframes in range(myallimages_sub.shape[-1]):
    # extract a single frame
    myimage = np.squeeze(np.float32(myallimages_sub[:,:,:,iframes]))
    
    # seperate calibration arrays into RGB
    I_exp = np.transpose(myimage, (2,0,1)) # Assuming CXY 
    R_exp = np.squeeze(I_exp[0,:,:])
    G_exp = np.squeeze(I_exp[1,:,:])
    B_exp = np.squeeze(I_exp[2,:,:])

    # estimate minimal norm solution 
    myinitopd = tf_jammin.findOPD(I_exp,R_map,G_map,B_map,mulfac)
    #plt.imshow(myinitopd), plt.colorbar(), plt.show()
    OPD_mask = np.ones(myinitopd.shape) # we don't want a mask here
    
    # This has to be assigned for each frame inidividually
    sess.run(tf.assign(TF_opd, myinitopd))

    for i in range(Niter):
        # Alternating? - Better not! 
        #my_loss_l2,_ = sess.run([TF_mySqrError, TF_loss_l2], feed_dict={TF_lr:lr})
        #my_loss_tv,_ = sess.run([TF_myTVError,TF_loss_TV], feed_dict={TF_lr:lr})
    
        # combined loss works best
        my_loss_tv,my_loss_l2,_ = sess.run([TF_myTVError,TF_mySqrError,TF_loss], feed_dict={TF_lr:lr, TF_lambda_TV:lambda_tv, TF_epsC:epsC,
                                           TF_R_exp: R_exp, TF_G_exp: G_exp, TF_B_exp: B_exp}) 
        
        #my_loss_tv,my_loss_l2,_ = sess.run([TF_myTVError,TF_mySqrError,TF_loss], feed_dict={TF_lr:lr}) 
        if(not np.mod(i,Ndisplay_text)):
            print("My Loss L2: @iter: "+str(i)+" is: "+str(my_loss_l2)+", My Loss TV: "+str(my_loss_tv))
        
        if(not np.mod(i, Ndisplay)):#is_debug):
            myopd_new = sess.run(TF_opd_masked)
            plt.imshow(myopd_new), plt.colorbar(), plt.show()
    

    myopd_new = sess.run(TF_opd_masked)
    myopd_old = myinitopd
    #myopd_new = myopd_new - np.min(myopd_new)
    
    
    plt.subplot(121)
    plt.imshow(myopd_old), plt.colorbar(fraction=0.046, pad=0.04), plt.title('Minimal Norm solution')
    plt.subplot(122)
    plt.imshow(myopd_new), plt.colorbar(fraction=0.046, pad=0.04), plt.title('Gradient Descent + TV')
    plt.figure()
    plt.title('Crosssection through the center')
    plt.plot(myopd_old[:,myopd_old.shape[1]//2])
    plt.plot(myopd_new[:,myopd_old.shape[1]//2]), plt.show()
    
    plt.imsave('myopd_old.tif', myopd_old)
    plt.imsave('myopd_new.tif', myopd_new)
    plt.imsave('myrgb_raw.tif', myimage)
    
    # save images for later 
    allframes_new.append(myopd_old)
    allframes_old.append(myopd_new)
    allframes_raw.append(myimage)

plt.imshow(myimage)

tif.imsave('tmp_new.tif', np.float32(np.array(allframes_new)))
tif.imsave('tmp_old.tif', np.float32(np.array(allframes_old)))
# tif.imsave('tmp_raw.tif', np.float32(np.array(allframes_raw)))