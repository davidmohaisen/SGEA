import networkx as nx
# import matplotlib.pyplot as plt
import os
import sys
import pygraphviz
import numpy as np


base = "/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/Graphs/"
families = ["Benign/","Gafgyt/","Mirai/","Tsunami/"]

for l in range(len(families)):

    print("-----------------------------------------")
    print("-----------------------------------------")
    print(families[l])
    counter = 0
    array = []
    directory = base + families[l]

    for files in os.listdir(directory):

        loc = directory + files
        g = ""
        try:
            g = nx.drawing.nx_agraph.read_dot(loc)
            t1 = len(g.nodes())
            t2 = len(g.edges())
            t = t1+t2
            array.append(t1)
        except:
            pass

    array.sort()

    min = array[0]
    max = array[len(array)-1]
    median = array[int(len(array)/2)]
    minFlag = -1
    maxFlag = -1
    medianFlag = -1

    directory = base + families[l]

    for files in os.listdir(directory):

        loc = directory + files
        g = ""
        try:
            g = nx.drawing.nx_agraph.read_dot(loc)
            t1 = len(g.nodes())
            t2 = len(g.edges())
            value = t1
            if value == max :
                if maxFlag==-1:
                    print("Max")
                    print(max)
                    print(loc)
                    maxFlag += 1
            if value == min :
                if minFlag==-1:
                    print("Min")
                    # minArray.append(t2)
                    # LocMinArray.append(loc)
                    print(min)
                    print(loc)
                    minFlag += 1
            if value == median :
                if medianFlag==-1:
                    # medianArray.append(t2)
                    # LocMedianArray.append(loc)
                    print("Median")
                    print(median)
                    print(loc)
                    medianFlag += 1
        except:
            pass
    print("-----------------------------------------")
    print("-----------------------------------------")
    # print("Min")
    # print(minArray)
    # print(LocMinArray)
    # print("Median")
    # print(medianArray)
    # print(LocMedianArray)
    # print("Max")
    # print(maxArray)
    # print(LocMaxArray)
