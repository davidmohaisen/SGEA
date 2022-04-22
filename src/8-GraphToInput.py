import networkx as nx
# import matplotlib.pyplot as plt
import os
import sys
import pygraphviz
import numpy as np
from shutil import copyfile
import random
import pickle
#IoT malware features
# FamilyNames = ["Benign","Gafgyt","Mirai","Tsunami"]
FamilyNames = ["Gafgyt"]


for l in range(len(FamilyNames)):
    directory = "/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/Graphs/"+FamilyNames[l]+"/"
    AllString = []
    counter = 0
    for files in os.listdir(directory):
        print(files)
        nodes_density = []
        loc = directory + files
        g = ""
        try:
            g = nx.drawing.nx_agraph.read_dot(loc)
            g = g.to_undirected()
        except:
            print("Passed Sample")
            pass
        if g!= "" :
            #### Start from here ####
            nodes = list(nx.nodes(g))
            edges = list(nx.edges(g))
            if len(nodes) <= 100 and counter < 1000:
                strToAdd = "t # "+str(counter)+"\n"
                for i in range(len(nodes)):
                    strToAdd += "v "+str(i)+" 0\n"
                for i in range(len(edges)):
                    strToAdd += "e "+str(nodes.index(edges[i][0]))+" "+str(nodes.index(edges[i][1]))+" 0\n"

                # csv_out = '/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/GraphTextPresentation/'+FamilyNames[l]+'.data'
                csv_out = '/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/GraphTextPresentation/'+FamilyNames[l]+'.dataSamples'
                with open(csv_out, "a") as output:
                    output.write(strToAdd)
                counter+=1
    # csv_out = '/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/GraphTextPresentation/'+FamilyNames[l]+'.data'
    csv_out = '/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/GraphTextPresentation/'+FamilyNames[l]+'.dataSamples'
    with open(csv_out, "a") as output:
        output.write("t # -1")
