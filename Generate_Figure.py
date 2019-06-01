#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:42:49 2019

@author: bene
"""

import matplotlib.pyplot as plt
import tifffile as tif
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rc('figure',  figsize=(14, 10))
mpl.rc('image', cmap='gray')

filepath_qphase = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/Jamin.Lebedeff/DOCUMENTS/PAPER/FIGURES/QP_HeLa_2019_05_09_v0.tif'
filepath_JL = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/Jamin.Lebedeff/DOCUMENTS/PAPER/FIGURES/JL_TV_50_eps_100_2019-03-22 17.35.50.tif'
filepath_JL_old = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/Jamin.Lebedeff/DOCUMENTS/PAPER/FIGURES/JL_TV_50_eps_100_2019-03-22 17.35.50_old.tif'

im_QP = tif.imread(filepath_qphase)
im_JL = tif.imread(filepath_JL)
im_JL_old = tif.imread(filepath_JL_old)

plt.figure()
plt.subplot(131)
ax = plt.gca()
im = ax.imshow(im_QP*(im_QP>0), cmap='gray'), plt.colorbar(), plt.axis('off')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)


plt.subplot(132), plt.imshow(im_JL, cmap='gray'), plt.colorbar(), plt.axis('off')
plt.subplot(133), plt.imshow(im_JL_old, cmap='gray'), plt.colorbar(), plt.axis('off')
plt.show()



import matplotlib.pyplot as plt


fig, (ax, ax2, cax) = plt.subplots(ncols=3,figsize=(5.5,3), 
                  gridspec_kw={"width_ratios":[1,1, 0.05]})
fig.subplots_adjust(wspace=0.3)
im  = ax.imshow(np.random.rand(11,8), vmin=0, vmax=1)
im2 = ax2.imshow(np.random.rand(11,8), vmin=0, vmax=1)
ax.set_ylabel("y label")

fig.colorbar(im, cax=cax)

plt.show()

