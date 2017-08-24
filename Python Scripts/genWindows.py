# Program to generate windows

import scipy as sp
import scipy.io as sio
from math import sqrt
import time

# Load Parameters
params       = sio.loadmat('/home/dell/Desktop/sarvaswa/objectness-release-v2.2/Trial_Pascal/testSet/exp1/salprop-v1.0/matpy/params.mat')
matpyfiles   = params['params'][0]['matpyfiles'][0]
windowsFile  = matpyfiles['windowsFile'][0][0][0]
salmapFile   = matpyfiles['salmapFile'][0][0][0]

iou      = params['params'][0]['windows'][0]['iou'][0][0][0][0]
scales   = params['params'][0]['windows'][0]['scale'][0][0][0]
aspRatio = params['params'][0]['windows'][0]['aspRatio'][0][0][0]

# Initializing and loading files
salmap = sio.loadmat(salmapFile)
salmap = salmap['salmap']
windows = []
rows,cols = salmap.shape[0],salmap.shape[1]
imgArea  = rows*cols
stepFactor = ((1-iou)/(1+iou))

# Generate Windows
for scale in scales:
    winArea = scale*imgArea
    for ar in aspRatio:
        width  = int(sp.round_(sqrt(winArea*ar)))
        height = int(sp.round_(sqrt(float(winArea)/ar)))
        stepH  = int(sp.round_(width*stepFactor))
        stepV  = int(sp.round_(height*stepFactor))
        for rmin in xrange(0,rows-height+1,stepV):
            for cmin in xrange(0,cols-width+1,stepH):
                rmax = min(rows-1,rmin+height)
                cmax = min(cols-1,cmin+width)
                windows.append([cmin,rmin,cmax,rmax])

windows = sp.array(windows)+1
sio.savemat(windowsFile, mdict={'windows':windows})
