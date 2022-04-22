import networkx as nx
import os
import sys
import pygraphviz
import numpy as np
from shutil import copyfile
import random
import pickle
#IoT malware features
FamilyNames = ["Benign","Gafgyt","Mirai","Tsunami"]


for l in range(len(FamilyNames)):
    directory = "../"+FamilyNames[l]+"/"
    AllString = []
    counter = 0
    for files in os.listdir(directory):
        print(files)
        nodes_density = []
        loc = directory + files
        g = ""
        try:
            g = nx.drawing.nx_agraph.read_dot(loc)
        except:
            print("Passed Sample")
            pass
        if g!= "" :
            #### Start from here ####
            node_cnt = len(list(nx.nodes(g)))
            edge_cnt = len(list(nx.edges(g)))
            avg_shortest_path = ""
            shortest_path = []
            closeness = []
            diameter = 0
            radius = 0
            current_flow_closeness = ""
            try:
                avg_shortest_path = nx.average_shortest_path_length(g)
                shortest_path = nx.shortest_path(g)
                closeness = nx.algorithms.centrality.closeness_centrality(g)
                shortest_betweenness = nx.algorithms.centrality.betweenness_centrality(g)
                degree_centrality = nx.algorithms.centrality.degree_centrality(g)
                density = nx.density(g)

            except:
                print("Unexpected error:", loc)
            sp_length = []
            for i in shortest_path:
                sp_length.append(shortest_path[i])
            shortestPathsArray = []
            for i in range(len(sp_length)):
                for x in sp_length[i] :
                    if (len(sp_length[i][x])-1)==0 :
                        continue
                    shortestPathsArray.append((len(sp_length[i][x])-1))

            if (len(shortestPathsArray))== 0 :
                counter += 1
                continue

            maxShortestPath = np.max(shortestPathsArray)
            minShortestPath = np.min(shortestPathsArray)
            meanShortestPath = np.mean(shortestPathsArray)
            medianShortestPath = np.median(shortestPathsArray)
            stdShortestPath = np.std(shortestPathsArray)
            closeness_list = list(closeness.values())
            betweenness_list = list(shortest_betweenness.values())
            degree_list = list(degree_centrality.values())
            out = str(l)+','+files+','+str(np.max(degree_list))+','+str(np.min(degree_list))+','+str(np.mean(degree_list))+','+str( np.median(degree_list))+','+str(np.std(degree_list))+','+str(np.max(betweenness_list))+','+str(np.min(betweenness_list))+','+str(np.mean(betweenness_list))+','+str( np.median(betweenness_list))+','+str(np.std(betweenness_list))+','+str(np.max(closeness_list))+','+str(np.min(closeness_list))+','+str(np.mean(closeness_list))+','+str( np.median(closeness_list))+','+str(np.std(closeness_list))+','+str(maxShortestPath)+','+str(minShortestPath)+','+str(meanShortestPath)+','+str(medianShortestPath)+','+str(stdShortestPath)+','+str(node_cnt)+','+str(edge_cnt)+','+str(density)+'\n'
            csv_out = '../Features.csv'
            with open(csv_out, "a") as output:
                output.write(out)
            counter+=1

    print(counter)
