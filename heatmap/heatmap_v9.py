import json
import os
import cv2
import numpy as np
import sys
import random
import matplotlib
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from matplotlib.image import NonUniformImage
from scipy.signal import medfilt

from matplotlib import pyplot as plt
import base64
from scipy.stats import kde
import io
from io import StringIO
from PIL import Image
import time


def _ploat_heatmap(x,y,s,height,width,bins=1000):
    heatmap,xedges,yedges = np.histogram2d(x ,y ,bins=bins, range=[[0,width],[0,height]])
    heatmap = gaussian_filter(heatmap,sigma=s)
    # print('heatmap------',heatmap)
    extent = [xedges[0],xedges[-1],yedges[0],yedges[-1]]
    return heatmap.T, extent

def heatmap(img1,centroid,height, width):
    time_stamp = time.time()
    # cv2.imshow('img',img)
    
    narr = np.array(centroid)
    print(narr)
    x,y = narr.T
    # height, width = img.shape[:2]
    img, extent = _ploat_heatmap(x,y,32,height, width)
    fig = plt.figure(frameon=False)
    # cv2.imshow('fig',fig)
    ax = plt.Axes(fig,[0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img,extent=extent,cmap=cm.jet,aspect='auto')
    heatmap_path = os.path.join('hmp_100per_intencity/hmp_{}.png'.format(time_stamp))
    fig.savefig(heatmap_path)
    alpha = 0.5
    heatmap_img = cv2.imread(heatmap_path)
    if heatmap_img.shape[:2] != img1.shape[:2]:
        heatmap_img = cv2.resize(heatmap_img, (width, height))
    print('heatmap img shape',heatmap_img.shape)
    print('img shape',img.shape)
    # heatmap_img = cv2.resize(heatmap_img,(width,height))
    blended_img = cv2.addWeighted(heatmap_img,alpha, img1, 1-alpha, 0 )
    overlayed_img_path = os.path.join("./heatmap_on_img.jpg")
    print("saving overlayed heatmap: {}".format(overlayed_img_path))
    # cv2.imwrite(overlayed_img_path,blended_img)
    # img = plt.imread("heatmap_on_img.jpg")
    print(img.shape)
    return blended_img

    