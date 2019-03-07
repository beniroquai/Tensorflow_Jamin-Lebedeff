import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py 
from mpl_toolkits.mplot3d.axes3d import Axes3D
import tf_spline as tf_spline
#%load_ext autoreload
#%autoreload 2



'''CODE STARTS HERE'''
# Here we try to do some kind of phase unwrapping for quantitative Phase images coming from Jamin Lebedeff Microscope 
mymatfile = './JL_tensorflow.mat'
matname = 'myParameter'

polydegree = 17
niterations = 5000
ndisplay = 100
lr = 1

#load system data; new MATLAB v7.3 Format! 
mat_matlab_data = h5py.File(mymatfile, 'r')
B_map = np.squeeze(np.array(mat_matlab_data['B_mat']))
R_map = np.squeeze(np.array(mat_matlab_data['R_mat']))
G_map = np.squeeze(np.array(mat_matlab_data['G_mat']))
OPD_map = np.array(mat_matlab_data['OPD_mat'])


# Display the curves 
plt.figure()
plt.title('Curves for RGB')
plt.plot(OPD_map,R_map, 'x')
plt.plot(OPD_map,B_map, 'x')
plt.plot(OPD_map,G_map, 'x')
plt.legend(['R', 'G','B'])
plt.show()

# now attempting to fit the curves
# define parameters 
normfac = np.mean(OPD_map)
R_mean = np.mean(R_map)
OPD_map = OPD_map/normfac

init_coeff_R = np.zeros((polydegree)); init_coeff_R[0] = np.mean(R_map); init_coeff_R[1] = .1; init_coeff_R[3] = 150/4
init_coeff_G = np.zeros((polydegree)); init_coeff_G[0] = np.mean(G_map); init_coeff_G[1] = .1; init_coeff_G[3] = 150/4
init_coeff_B = np.zeros((polydegree)); init_coeff_B[0] = np.mean(B_map); init_coeff_B[1] = .1; init_coeff_B[3] = 150/4
TF_par_R = tf.Variable(init_coeff_R)
TF_par_G = tf.Variable(init_coeff_G)
TF_par_B = tf.Variable(init_coeff_B)
TF_exp_R = tf.constant(np.squeeze(R_map))
TF_exp_G = tf.constant(np.squeeze(G_map))
TF_exp_B = tf.constant(np.squeeze(B_map))
TF_OPD = tf.constant(np.squeeze(OPD_map))

# make predictions
TF_pred_R = tf_spline.myPolyFunc(TF_OPD, TF_par_R)
TF_pred_G = tf_spline.myPolyFunc(TF_OPD, TF_par_G)
TF_pred_B = tf_spline.myPolyFunc(TF_OPD, TF_par_B)

# Define error for curve fitting
myErrR = tf.reduce_sum(tf.square(TF_pred_R - TF_exp_R)) + tf.abs(TF_par_R[1])*1000
myErrG = tf.reduce_sum(tf.square(TF_pred_G - TF_exp_G)) + tf.abs(TF_par_G[1])*1000
myErrB = tf.reduce_sum(tf.square(TF_pred_B - TF_exp_B)) + tf.abs(TF_par_B[1])*1000

# Define optimizers 
#TF_opt_R = tf.train.GradientDescentOptimizer(learning_rate = lr)
TF_opt_R = tf.train.AdamOptimizer(learning_rate = lr)
TF_loss_R = TF_opt_R.minimize(myErrR)
TF_opt_G = tf.train.AdamOptimizer(learning_rate = lr)
TF_loss_G = TF_opt_G.minimize(myErrG)
TF_opt_B = tf.train.AdamOptimizer(learning_rate = lr)
TF_loss_B = TF_opt_B.minimize(myErrB)



#%% Scipy spline fit 
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

x = OPD_map # np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/8)
y = R_map #np.sin(x)
t, c, k = interpolate.splrep(x[0::5], R_map[0::5], s=0) #  containing the knot-points,  , the coefficients  and the order of the spline.
xnew = np.squeeze(OPD_map)#np.arange(0, 2*np.pi, np.pi/50)
R_map_new = tf_spline.bspleval(np.squeeze(OPD_map), t, c, k, debug=False)

#%%
import tf_spline as tf_spline
OPD_map_tf = tf.constant(np.squeeze(OPD_map))


R_map_new_tf = tf_spline.bspleval_tf(OPD_map_tf, t, c, k, debug=False)



sess = tf.Session()
sess.run(tf.global_variables_initializer())

# op to write logs to Tensorboard
logs_path = './logs'
#summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

import time
t1 = time.time()
R_map_new_tf_eval =sess.run(R_map_new_tf)
print(time.time()-t1)


plt.figure()
plt.plot(x, y, '-')
plt.plot(xnew, myy, 'x')
plt.legend(['Linear', 'Cubic Spline Numpy', 'True'])
plt.title('Cubic-spline interpolation')
plt.show()

#%% fit spline
if(1):
    x = np.squeeze(OPD_map[0::7]) #np.arange(10.0)
    x = x-np.min(x)
    #x = x[0:20]
    x = x.astype(np.float32, copy=False)
    y = R_map[0::7]#np.sin(x)#*x**2+x
    #y = y[0:20]
    y = y.astype(np.float32, copy=False)
    xs = np.linspace(0,1.7,100)#np
    xs = xs.astype(np.float32, copy=False)
    
    y_tensor = tf.constant(y)
    xs_tensor = tf.constant(xs)
    ys_tensor = tf_spline.cubic_spline(x, y_tensor, xs_tensor)
    
    gradients_tensor = tf.gradients(ys_tensor, y_tensor)
    
    session = tf.Session()
    res = session.run([ys_tensor, gradients_tensor])
    ys = res[0]
    
    print("Num data points:", len(xs)) # 80000 data points
    print("Gradients:")
    print(res[1])
    
    plt.figure(figsize=(6.5, 4))
    plt.plot(x, y, label='data')
    plt.plot(xs, ys, 'x', label='spline')
    
    #plt.xlim(-0.5, 9.5)
    plt.legend(loc='lower right', ncol=2)
    plt.show()
    
#%% fit function using scipy
from scipy.optimize import curve_fit
def func(x,a,b,c,d,e,f,g,h,i,j):
#    return a + np.sin(b*x+c)*d + e*x + f*x**2 + g*x**3 + h*x**4 + np.cos(x*i+j)*k + l*x**5 #+ np.exp(l*(x+m))*n
    return a + b*np.exp(-c*(x-d))*(np.cos(e*x+f)) + g*x + h*x**2 + i*x**3 + j*x**4#*g+np.sin(h*x+i)*j)
#y(t)=A\cdot e^{{-\lambda t}}\cdot (\cos(\omega t+\phi )+\sin(\omega t+\phi ))
popt, pcov = curve_fit(func, np.squeeze(OPD_map), R_map)

a,b,c,d,e,f,g,h,i,j = popt

R_map_fit = func(OPD_map,a,b,c,d,e,f,g,h,i,j )
plt.plot(OPD_map, R_map)
plt.plot(OPD_map, R_map_fit), plt.show()

#%% start optimization part here
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(niterations):
    #plt.bar(range(sess.run(TF_par_R).shape[0]), sess.run(TF_par_R)), plt.show()
    errR, _ = sess.run([myErrR, TF_loss_R])
    errG, _ = sess.run([myErrG, TF_loss_G])
    errB, _ = sess.run([myErrB, TF_loss_B])    
    if(not np.mod(i,ndisplay)):
        #print("Error R at "+str(i)+" is: "+str(errR))
        #print("Error G at "+str(i)+" is: "+str(errG))
        #print("Error B at "+str(i)+" is: "+str(errB))
    #print(sess.run(TF_par_R))

        
        # evaluate result and compare fitted line to measured one
        plt.figure()
        plt.title('Curves for R')
        plt.plot(OPD_map,R_map, 'x')
        plt.plot(OPD_map,tf_spline.myPolyFunc(np.squeeze(OPD_map), sess.run(TF_par_R)), 'x')
        plt.legend(['R', 'R_fit'])
        plt.show()
        
        plt.figure()
        plt.title('Curves for G')
        plt.plot(OPD_map,G_map, 'x')
        plt.plot(OPD_map,tf_spline.myPolyFunc(np.squeeze(OPD_map), sess.run(TF_par_G)), 'x')
        plt.legend(['G', 'G_fit'])
        plt.show()
        
        plt.figure()
        plt.title('Curves for B')
        plt.plot(OPD_map,B_map, 'x')
        plt.plot(OPD_map,tf_spline.myPolyFunc(np.squeeze(OPD_map), sess.run(TF_par_B)), 'x')
        plt.legend(['B', 'B_fit'])
        plt.show()

plt.bar(range(sess.run(TF_par_R).shape[0]), sess.run(TF_par_R)), plt.show(); print(sess.run(TF_par_R))
plt.bar(range(sess.run(TF_par_G).shape[0]), sess.run(TF_par_G)), plt.show(); print(sess.run(TF_par_G))
plt.bar(range(sess.run(TF_par_B).shape[0]), sess.run(TF_par_B)), plt.show(); print(sess.run(TF_par_B))


R_fitted = tf_spline.myPolyFunc(np.squeeze(OPD_map), sess.run(TF_par_R))
G_fitted = tf_spline.myPolyFunc(np.squeeze(OPD_map), sess.run(TF_par_G))
B_fitted = tf_spline.myPolyFunc(np.squeeze(OPD_map), sess.run(TF_par_B))

#%%--- 3D Plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.plot(R_fitted, G_fitted, B_fitted, color='y', lw=2, label='Fitted Curve')
ax.plot(R_map, G_map, B_map, color='b', lw=2, label='RAW Curve')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_aspect('equal', 'datalim')



#%%


lkjh

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
    
    
    
