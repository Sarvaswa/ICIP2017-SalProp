# Script to predict edge map

# Importing Libraries
from pystruct.utils import SaveLogger
import scipy.io as sio

# Getting parameters
params = sio.loadmat('/home/dell/Desktop/sarvaswa/objectness-release-v2.2/Trial_Pascal/testSet/exp1/salprop-v1.0/matpy/params.mat')
matpyfiles   = params['params'][0]['matpyfiles'][0]
featureFile  = matpyfiles['featureFile'][0][0][0]
edgesFile    = matpyfiles['edgesFile'][0][0][0]
predictFile  = matpyfiles['predictFile'][0][0][0]
modelFile    = params['params'][0]['modelFileCRF'][0][0]

# Loading required Files
modelLogger = SaveLogger(modelFile)
crf = modelLogger.load()
feat = sio.loadmat(featureFile)
feat = feat['feat']
edges = sio.loadmat(edgesFile)
edges = edges['edges'] - 1

# Make Prediction
inData = [(feat,edges)]
prediction = crf.predict(inData)

# Save Prediction;
sio.savemat(predictFile, mdict = {'prediction':prediction})