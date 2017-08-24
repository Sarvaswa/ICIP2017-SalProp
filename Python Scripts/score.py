#Edge Straddling Scoring

import os
import scipy as sp
import scipy.io as sio
from math import sqrt,exp
import cv2

# Function to Generate Windows
def genBoxCompMat(box):
    cmin,rmin,cmax,rmax = box
    return [rmin,-rmax,rmin,-rmax,rmin,-rmax,rmin,-rmax,cmin,-cmax,cmin,-cmax,cmin,-cmax,cmin,-cmax]

# Function to compute Score for bounding box
def getBoxScore(box,edgeCompMat,salLen,area):
    try:
        boxCompMat = genBoxCompMat(box)
        comp = (edgeCompMat>=boxCompMat)
        score = sp.sum(comp,axis=1)
        inIdxs = score==16
        #otherEdges = comp[score<16,:]
        #ptSum = (sp.sum(otherEdges[:,[0,1,8,9]],axis=1),sp.sum(otherEdges[:,[2,3,10,11]],axis=1),sp.sum(otherEdges[:,[4,5,12,13]],axis=1),sp.sum(otherEdges[:,[6,7,14,15]],axis=1))
        #ptAll = sp.array(ptSum).transpose()
        #ptIn  = sp.any(ptAll==4,axis=1)
        #stradScore = 1./(1+exp(-1*sum(ptIn)))
        inScore = sum(salLen[inIdxs])
        boxScore = (float(inScore)/area**0.5)
    except:
        print box,area,inScore
        raise('ValueError')
    return boxScore

# Compute area of the box
def getBoxArea(box):
     return ((box[3]-box[1]+1)*(box[2]-box[0]+1))

# Refine Boxes
def refineBoxes(sortBoxes,edgeCompMat,salLen,iou,sizeImg):
##    print 'refining'
    # Boxes to be refined    
    rows,cols = sizeImg
    w = sortBoxes[:,2]-sortBoxes[:,0]
    h = sortBoxes[:,3]-sortBoxes[:,1]
    area  = sortBoxes[:,4]
    score = sortBoxes[:,5]
    numBoxes = sp.shape(sortBoxes)[0]
    iouStepConv = (1-iou)/(1+iou)

    # Start Refinement
    for i in xrange(numBoxes):
        
        box = sortBoxes[i,0:4]
        Box = box.copy()
        
        currScore = score[i]
##        print i,currScore
##        print i
        cStep,rStep = w[i]*iouStepConv,h[i]*iouStepConv
        while 1:
            
            cStep,rStep = int(cStep/2),int(rStep/2)
            if (cStep<=2 and rStep<=2):
##                if sp.any(sortBoxes[i,:]!=Box):
##                    print 'refined'
                sortBoxes[i,0:4] = Box
                sortBoxes[i,5]   = currScore
                break
            
            # Search at rmin
            Box=box.copy(); Box[1]=max(Box[1]-rStep,0);
            boxArea  = getBoxArea(Box)
            newScore = getBoxScore(Box,edgeCompMat,salLen,boxArea)
##            print '-',i,box,'rmin-out ',rStep,Box,currScore,newScore
            if (newScore<=currScore):
                Box = box.copy(); Box[1]=min(Box[1]+rStep,rows-1);
                boxArea  = getBoxArea(Box) 
                newScore=getBoxScore(Box,edgeCompMat,salLen,boxArea)
##                print '-',i,box,'rmin-in ',rStep,Box,currScore,newScore
            if (newScore>currScore):
##                print 'assigned'
                box = Box.copy()
                currScore = newScore

            # Search at rmax
            Box = box.copy(); Box[3]=max(Box[3]-rStep,0);
            boxArea = getBoxArea(Box)            
            newScore=getBoxScore(Box,edgeCompMat,salLen,boxArea)
##            print '-',i,box,'rmax-in ',rStep,Box,currScore,newScore
            if (newScore<=currScore):
                Box = box.copy(); Box[3]=min(Box[3]+rStep,rows-1);
                boxArea  = getBoxArea(Box)
                newScore=getBoxScore(Box,edgeCompMat,salLen,boxArea)
##                print '-',i,box,'rmax-out ',rStep,Box,currScore,newScore
            if (newScore>currScore):
##                print 'assigned'
                box = Box.copy()
                currScore = newScore
                
            # Search at cmin
            Box = box.copy(); Box[0]=max(Box[0]-cStep,0);
            boxArea = getBoxArea(Box)
            newScore=getBoxScore(Box,edgeCompMat,salLen,boxArea)
##            print '-',i,box,'cmin-out ',cStep,Box,currScore,newScore
            if (newScore<=currScore):
                Box = box.copy(); Box[0]=min(Box[0]+cStep,cols-1);
                boxArea  = getBoxArea(Box)
                newScore=getBoxScore(Box,edgeCompMat,salLen,boxArea)
##                print '-',i,box,'cmin-in ',cStep,Box,currScore,newScore
            if (newScore>currScore):
##                print 'assigned'
                box = Box.copy()
                currScore = newScore
                
            # Search at cmax
            Box = box.copy(); Box[2]=max(Box[2]-cStep,0);
            boxArea = getBoxArea(Box)
            newScore=getBoxScore(Box,edgeCompMat,salLen,boxArea)
##            print '-',i,box,'cmax-in ',cStep,Box,currScore,newScore
            if (newScore<=currScore):
                Box = box.copy(); Box[2]=min(Box[2]+cStep,cols-1);
                boxArea  = getBoxArea(Box)
                newScore=getBoxScore(Box,edgeCompMat,salLen,boxArea)
##                print '-',i,box,'cmax-out ',cStep,Box,currScore,newScore
            if (newScore>currScore):
##                print 'assigned'
                box = Box.copy()
                currScore = newScore
    
    sortBoxes = sortBoxes[(-sortBoxes[:,5]).argsort()]
    return sortBoxes
    
# Function to generate edge comparison matrix
def genEdgeCompMat(connComp,rows):
    numComps  = sp.shape(connComp)[0]
    edgeCompMat = sp.zeros((numComps,16))
    for i in xrange(numComps):
        comp = connComp[i].astype(sp.int32)
        r = comp%rows
        c = comp/rows + 1
        r[r==0] = rows
        subs = sp.concatenate((r,c),axis=1)-1

        r_cmin,cmin = subs[sp.argmin(c),:]
        rmin,c_rmin = subs[sp.argmin(r),:]
        r_cmax,cmax = subs[sp.argmax(c),:]
        rmax,c_rmax = subs[sp.argmax(r),:]

        edgeCompMat[i,:] = [r_cmin,-r_cmin,rmin,-rmin,r_cmax,-r_cmax,rmax,-rmax,cmin,-cmin,c_rmin,-c_rmin,cmax,-cmax,c_rmax,-c_rmax]
    return edgeCompMat

params       = sio.loadmat('/home/dell/Desktop/sarvaswa/objectness-release-v2.2/Trial_Pascal/testSet/exp1/salprop-v1.0/matpy/params.mat')
matpyfiles   = params['params'][0]['matpyfiles'][0]
connCompFile = matpyfiles['connCompFile'][0][0][0]
salmapFile   = matpyfiles['salmapFile'][0][0][0]
emapClusFile = matpyfiles['emapClusFile'][0][0][0]
windowsFile  = matpyfiles['windowsFile'][0][0][0]
scoreFile    = matpyfiles['scoreFile'][0][0][0]

#Load Clustered Edge Map and Windows
salmap   = sio.loadmat(salmapFile)
salmap   = salmap['salmap']
emapClus = sio.loadmat(emapClusFile)
emapClus = emapClus['emapClus']
windows  = sio.loadmat(windowsFile)
windows  = windows['windows']-1
windows  = windows.astype(sp.int32)
connComp = sio.loadmat(connCompFile)
connComp = connComp['connComp']['PixelIdxList'][0][0][0]

rows,cols = sp.shape(emapClus)

# Generate Edge Comparison Matrix
edgeCompMat = genEdgeCompMat(connComp,rows)
    
# Extract Label Information
numLabels = sp.amax(emapClus)
labels = range(1,numLabels+1)

# Compute saliency and length of each label
salLen = sp.zeros((numLabels,1))
for i in xrange(numLabels):
    #length = sp.count_nonzero(emapClus==labels[i])
    salVals = salmap[emapClus==labels[i]]
    sal= sp.amax(salVals)
    salLen[i] = sal;#*length

#Compute Area of each Window
areaWin = ((windows[:,2]-windows[:,0]+1)*(windows[:,3]-windows[:,1]+1))
numWindows = windows.shape[0]

#Compute Window Scores
score = sp.zeros((numWindows))
for i in xrange(numWindows):
    box = windows[i,:]
    score[i] = getBoxScore(box,edgeCompMat,salLen,areaWin[i])

sortBoxes = sp.hstack((windows,sp.reshape(areaWin,(numWindows,1)),sp.reshape(score,(numWindows,1))))
sortBoxes = sortBoxes[(-sortBoxes[:,5]).argsort()]
maxScore  = max(sortBoxes[:,5])
meanScore = sp.mean(sortBoxes[:,5])
##print meanScore,maxScore
thresh = 0.6; iou = 0.65;
refBoxes = sortBoxes[sortBoxes[:,5]>thresh*maxScore,:]
##print refBoxes.shape
numRefBoxes = sp.shape(refBoxes)[0]
boxes  = refineBoxes(refBoxes,edgeCompMat,salLen,iou,[rows,cols])
sortBoxes[0:numRefBoxes,:] = boxes;
scores = sp.hstack((sortBoxes[:,0:4],sp.reshape(sortBoxes[:,5],(numWindows,1))))

##print 'max score: ',sp.amax(score)
##box = windows[sp.argmax(score),:]
##boxArea = (box[3]-box[1]+1)*(box[2]-box[0]+1)
##print 'window: ',box
##print 'calculated score: ',getBoxScore(box,edgeCompMat,salLen,boxArea)

#Saving Scores
sio.savemat(scoreFile, mdict={'scores':scores})