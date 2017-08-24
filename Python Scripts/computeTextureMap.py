#Script to run a 17D filter on an image to compute Texture Map
from scipy import dstack, average
import scipy.io as sio
import scipy.ndimage as nd
from cv2 import imread, cvtColor, COLOR_BGR2LAB, blur

matpyDir = '/home/dell/Desktop/sarvaswa/objectness-release-v2.2/Trial_Pascal/testSet/exp1/salprop-v1.0/matpy/'
imgFile = matpyDir + 'imgFile.png'
colorTextureFile = matpyDir + 'colorTextureFile.mat'

#Loading Image and defining parameters
img = imread(imgFile)
imgLab = cvtColor(img,COLOR_BGR2LAB)
L = imgLab[:,:,0]
sigma = 0.5

#TEXTURE FEATURES
#LoG Computation on L channel of LAB Image
LoG1 = nd.gaussian_laplace(L,sigma)
LoG2 = nd.gaussian_laplace(L,2*sigma)
LoG3 = nd.gaussian_laplace(L,4*sigma)
LoG  = dstack((LoG1,LoG2,LoG3))
LoG  = average(LoG,axis=2)

#Gaussian Filter Computation on L,A,B channels of LAB Image
G1 = nd.gaussian_filter(L,sigma)
G2 = nd.gaussian_filter(L,2*sigma)
G3 = nd.gaussian_filter(L,4*sigma)

#DoG Computation on L channel of LAB Image
DoG1 = G2 - G1
DoG2 = G3 - G2
DoG  = dstack((DoG1,DoG2))
DoG  = average(DoG,axis=2)

#Generate Features using above computed matrices
feat = dstack((DoG,LoG))

#Save ColorTextureFile
sio.savemat(colorTextureFile, mdict={'feat':feat})