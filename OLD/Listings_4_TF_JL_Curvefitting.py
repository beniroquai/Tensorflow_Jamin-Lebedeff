import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py 
from mpl_toolkits.mplot3d.axes3d import Axes3D
import tf_spline as tf_spline
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import tf_spline as tf_spline
import time
#%load_ext autoreload
#%autoreload 2



'''CODE STARTS HERE'''
# Here we try to do some kind of phase unwrapping for quantitative Phase images coming from Jamin Lebedeff Microscope 
mymatfile = './JL_tensorflow.mat'
matname = 'myParameter'

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

#%%
''' Scipy spline fit '''
x = OPD_map # np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/8)
y = R_map #np.sin(x)
t, c, k = interpolate.splrep(x[0::5], R_map[0::5], s=0) #  containing the knot-points,  , the coefficients  and the order of the spline.
xnew = np.squeeze(OPD_map)#np.arange(0, 2*np.pi, np.pi/50)
R_map_new = tf_spline.bspleval(np.squeeze(OPD_map), t, c, k, debug=False)

#%%
''' Tensorflow spline fit '''
OPD_map_tf = tf.constant(np.squeeze(OPD_map))

R_map_new_tf = tf_spline.bspleval_tf(OPD_map_tf, t, c, k, debug=False)

# start session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# op to write logs to Tensorboard
logs_path = './logs'
#summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

# get the spline fitted curve
t1 = time.time()
R_map_new_tf_eval =sess.run(R_map_new_tf)
print(time.time()-t1)

#%%
''' Display the result'''
plt.figure()
plt.plot(x, y, '-')
plt.plot(xnew, R_map_new, 'R', '-')
plt.plot(xnew, R_map_new_tf_eval, 'G', '_')
plt.legend(['Linear', 'Cubic Spline Numpy', 'True'])
plt.title('Cubic-spline interpolation')
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

#%%fit polynomial 
p30 = np.poly1d(np.polyfit(np.squeeze(OPD_map), R_map, 10))
plt.imshow(p30(np.random.rand(100,100)))