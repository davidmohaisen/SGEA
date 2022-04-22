import networkx as nx
# import matplotlib.pyplot as plt
import os
import sys
import pygraphviz
import numpy as np
from shutil import copyfile
import random
import pickle
import keras

mainVisualizationVector = open('/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/FeatureVector/Features.csv')
mainVisualizationVector = mainVisualizationVector.readlines()
accs = []
z = []
x = []
y = []

for i in range(len(mainVisualizationVector)):
    z.append(mainVisualizationVector[i].split(','))
for i in range(len(z)):
    y.append(int(z[i][0]))
    x.append(z[i][2:])
for i in range(len(x)):
    for j in range(len(x[i])):
        x[i][j]=  float(x[i][j])

x=np.asarray(x)
y=np.asarray(y)
maxX = [0] * 23
for i in range(len(x)):
    for j in range(len(x[i])):
        if maxX[j] < x[i][j]:
            maxX[j] = x[i][j]
for i in range(len(x)):
    for j in range(len(x[i])):
        if maxX[j] != 0 :
            x[i][j] = x[i][j]/maxX[j]


taken = [0,0,0,0]
size = [0,0,0,0]
for i in range(len(y)):
    size[y[i]] += 1

xTrain = []
xTest = []
yTrain = []
yTest = []

for i in range(len(y)):
    if taken[y[i]] < 0.8*size[y[i]]:
        xTrain.append(x[i])
        yTrain.append(y[i])
    else :
        xTest.append(x[i])
        yTest.append(y[i])
    taken[y[i]]+=1

howMuchTestToken = [0,0,0,0]
howMuchTrainToken = [0,0,0,0]

for i in range(len(yTrain)):
    howMuchTrainToken[yTrain[i]]+=1
for i in range(len(yTest)):
    howMuchTestToken[yTest[i]]+=1
print(size)
print(howMuchTrainToken)
print(howMuchTestToken)

xTrain = np.asarray(xTrain)
xTest = np.asarray(xTest)
yTrain = np.asarray(yTrain)
yTest = np.asarray(yTest)

xTrain = xTrain.reshape((len(xTrain),23,1))
xTest = xTest.reshape((len(xTest),23,1))
yTrain = keras.utils.to_categorical(yTrain, 4)
yTest = keras.utils.to_categorical(yTest, 4)

print(xTrain.shape)
print(xTest.shape)
print(yTrain.shape)
print(yTest.shape)

x_trainFile = open('/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/FeatureVector/xTrain.pkl', 'wb')
pickle.dump(xTrain, x_trainFile)
x_trainFile = open('/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/FeatureVector/xTest.pkl', 'wb')
pickle.dump(xTest, x_trainFile)
x_trainFile = open('/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/FeatureVector/yTrain.pkl', 'wb')
pickle.dump(yTrain, x_trainFile)
x_trainFile = open('/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/FeatureVector/yTest.pkl', 'wb')
pickle.dump(yTest, x_trainFile)
