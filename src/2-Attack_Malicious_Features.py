import networkx as nx
# import matplotlib.pyplot as plt
import os
import sys
import pygraphviz
import numpy as np
import time

base = "/home/ahmed/Documents/Projects/IoT_Attack_Journal/GEA/TestList/"
saveBase = "/home/ahmed/Documents/Projects/IoT_Attack_Journal/GEA/Features/"
families = ["Benign/","Gafgyt/","Mirai/","Tsunami/"]
classes = ["Benign","Gafgyt","Mirai","Tsunami"]
size = ["Minimum","Median","Maximum"]
samples = ["/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/Graphs/Benign/_negvdi2_s.o.dot",
            "/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/Graphs/Benign/setopt.c.o.dot",
            "/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/Graphs/Benign/zip.o.dot",
            "/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/Graphs/Gafgyt/6ba789055c3a732de1b3cb1bcc0acc62.dot",
            "/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/Graphs/Gafgyt/6dd5461a19d9b23249509f524a344a0b.dot",
            "/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/Graphs/Gafgyt/d5d1ad0f90f64bc465807ae6d5980e1b.dot",
            "/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/Graphs/Mirai/fcee4183a0bf93e1e24fea34fd330b82.dot",
            "/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/Graphs/Mirai/ed8e5a6efa3a6e96769b5666edf436ba.dot",
            "/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/Graphs/Mirai/3b00eb01e109297e461f825e6a69e518.dot",
            "/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/Graphs/Tsunami/056867da245c85783a62fb662e4a323e.dot",
            "/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/Graphs/Tsunami/fde1712765d2a8d4ba51ba01fb975e6d.dot",
            "/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/Graphs/Tsunami/5506b22561aa7bad235a97335380fe71.dot"]


for l1 in range(len(classes)):

    for l2 in range(len(size)):
        print("___________________________________________")
        print(classes[l1],size[l2])
        timeFirst = time.time()
        counter = 0

        for l3 in range(len(families)):
            if classes[l1] in families[l3]:
                continue

            directory = base + families[l3]
            toMerge = samples[l1*3+l2]


            for files in os.listdir(directory):

                loc = directory + files
                g = ""
                try:
                    g1 = nx.drawing.nx_agraph.read_dot(loc)
                    g2 = nx.drawing.nx_agraph.read_dot(toMerge)
                    g = nx.compose(g1,g2)
                    n1 = (g1.nodes())
                    n2 = (g2.nodes())
                    xs = []
                    n1 = np.asarray(n1)
                    n2 = np.asarray(n2)
                    nEntry_Label = int(n1[0], 16)
                    nEntry_Label -= 12
                    g.add_node(hex(nEntry_Label))
                    g.add_edge(hex(nEntry_Label),n1[0])
                    g.add_edge(hex(nEntry_Label),n2[0])
                    nExit_Label = int(n1[len(n1)-1], 16)
                    nExit_Label += 12
                    g.add_node(hex(nExit_Label))
                    g.add_edge(n1[len(n1)-1],hex(nExit_Label))
                    g.add_edge(n2[len(n2)-1],hex(nExit_Label))
                except:
                    pass
                if g!= "" :
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
                    out = str(l3)+','+files+','+str(np.max(degree_list))+','+str(np.min(degree_list))+','+str(np.mean(degree_list))+','+str( np.median(degree_list))+','+str(np.std(degree_list))+','+str(np.max(betweenness_list))+','+str(np.min(betweenness_list))+','+str(np.mean(betweenness_list))+','+str( np.median(betweenness_list))+','+str(np.std(betweenness_list))+','+str(np.max(closeness_list))+','+str(np.min(closeness_list))+','+str(np.mean(closeness_list))+','+str( np.median(closeness_list))+','+str(np.std(closeness_list))+','+str(maxShortestPath)+','+str(minShortestPath)+','+str(meanShortestPath)+','+str(medianShortestPath)+','+str(stdShortestPath)+','+str(node_cnt)+','+str(edge_cnt)+','+str(density)+'\n'
                    csv_out = saveBase+families[l1]+size[l2]+'/features.csv'
                    with open(csv_out, "a") as output:
                        output.write(out)
        timeLast = time.time()  - timeFirst
        print(str(timeLast)+" seconds")
exit()
