import networkx as nx
# import matplotlib.pyplot as plt
import os
import sys
import pygraphviz
import numpy as np
from shutil import copyfile
import random
import pickle
import pygraphviz
from networkx.drawing.nx_agraph import write_dot
family = ["Benign","Gafgyt","Mirai","Tsunami"]

for l in range(len(family)):
    token = []
    print(family[l])
    f = open("/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/GraphVisualization/"+family[l]+"SubGraphs","r")
    lines = f.readlines()

    count = 0
    for i in range(len(lines)):
        if lines[i].count("t ")!= 0: # New Graph
            G=nx.Graph()
            nodesCount = 0
            edgesCount = 0
            while True:
                i += 1
                if lines[i].count("v ")!= 0:
                    G.add_node(nodesCount)
                    nodesCount += 1
                elif lines[i].count("e ")!= 0:
                    edgesCount += 1
                    line = lines[i]
                    line = line.split(" ")
                    G.add_edge(int(line[1]),int(line[2]))
                else:
                    break
            if [nodesCount,edgesCount] not in token:
                token.append([nodesCount,edgesCount])
                write_dot(G,"/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/GraphVisualization/"+family[l]+"/"+str(count)+".dot")
                count += 1
