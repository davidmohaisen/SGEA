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
    x.append(z[i][1])

families= ["Benign","Gafgyt","Mirai","Tsunami"]
base = "/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/Graphs/"
toAddress = "/home/ahmed/Documents/Projects/IoT_Attack_Journal/GEA/TestList/"

taken = [0,0,0,0]
size = [0,0,0,0]
for i in range(len(y)):
    size[y[i]] += 1



for i in range(len(y)):
    if taken[y[i]] >= 0.8*size[y[i]]:
        copyfile(base+families[y[i]]+"/"+x[i], toAddress+families[y[i]]+"/"+x[i])

    taken[y[i]]+=1
