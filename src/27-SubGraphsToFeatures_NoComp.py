from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import time
import logging
import os
from numpy import argmax
import pickle
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import DeepFool
from cleverhans.attacks import MadryEtAl
from cleverhans.loss import CrossEntropy
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import VirtualAdversarialMethod
from cleverhans.utils import grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import train, model_eval, tf_model_load, model_argmax
from cleverhans_tutorials.tutorial_models import ModelBasicDNN, ModelBasicCNN
from six.moves import xrange
from sklearn.model_selection import train_test_split,KFold
import keras
from cleverhans.attacks import SPSA
from cleverhans.attacks import SaliencyMapMethod
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
FLAGS = flags.FLAGS

LEARNING_RATE = .001
CW_LEARNING_RATE = .2
ATTACK_ITERATIONS = 1000



def mnist_tutorial_cw(train_start=0, train_end=60000, test_start=0,
                      test_end=10000, viz_enabled=True, nb_epochs=6,
                      batch_size=50, source_samples=10,
                      learning_rate=LEARNING_RATE,
                      attack_iterations=ATTACK_ITERATIONS,
                      model_path=os.path.join("models", "mnist"),
                      targeted=True):

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    x_train = open("/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/FeatureVector/xTrain.pkl","rb")
    u = pickle._Unpickler(x_train)
    u.encoding = 'latin1'
    x_train = u.load()

    x_test = open("/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/FeatureVector/xTest.pkl","rb")
    u = pickle._Unpickler(x_test)
    u.encoding = 'latin1'
    x_test = u.load()

    y_train = open("/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/FeatureVector/yTrain.pkl","rb")
    u = pickle._Unpickler(y_train)
    u.encoding = 'latin1'
    y_train = u.load()

    y_test = open("/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/FeatureVector/yTest.pkl","rb")
    u = pickle._Unpickler(y_test)
    u.encoding = 'latin1'
    y_test = u.load()


    img_rows, img_cols = x_train.shape[1:3]
    nb_classes = y_train.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols))
    xprime = x
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    nb_filters = 46

    # Define TF model graph
    model = ModelBasicDNN(nb_classes, nb_filters)
    # model = ModelBasicCNN(nb_classes, nb_filters)
    preds = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=0.1)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################
    rng = np.random.RandomState([2019, 5, 2])
    # check if we've trained before, and if we have, use that pre-trained model

    train_params = {
       'nb_epochs': 100,
       'batch_size': 32,
       'learning_rate': .001
    }

    train(sess, loss, x, y, x_train, y_train, args=train_params,
          save=os.path.exists("models"), rng=rng)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
    print('Test accuracy on clean test examples: {0}'.format(accuracy))



    org = open('/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/FeatureVector/Features.csv')
    org = org.readlines()

    z1 = []
    x1 = []
    y1 = []
    for i in range(len(org)):
        z1.append(org[i].split(','))
    for i in range(len(z1)):
        y1.append(int(z1[i][0]))
        x1.append(z1[i][2:])
    for i in range(len(x1)):
        for j in range(len(x1[i])):
            x1[i][j]=  float(x1[i][j])

    x1=np.asarray(x1)
    y1=np.asarray(y1)

    maxX = [0] * 23
    for i in range(len(x1)):
        for j in range(len(x1[i])):
            if maxX[j] < x1[i][j]:
                maxX[j] = x1[i][j]




    #IoT malware features
    FamilyNames = ["Benign","Gafgyt","Mirai","Tsunami"]
    base = "/home/ahmed/Documents/Projects/IoT_Attack_Journal/GEA/TestList/"
    saveBase = "/home/ahmed/Documents/Projects/IoT_Attack_Journal/GEA/Features/"
    families = ["Benign/","Gafgyt/","Mirai/","Tsunami/"]

    for l1 in range(len(FamilyNames)):
        print("___________________________________________")
        print(FamilyNames[l1])
        timeFirst = time.time()

        directory = base + families[l1]
        Misclassification = [0,0,0,0]
        TMisclassification = [0,0,0,0]
        All = 0
        for files in os.listdir(directory):
            All += 1
            # print(All)
            loc = directory + files
            for l in range(len(FamilyNames)):
                if l1 != l :
                    alreadyMisclass = 0
                    directory1 = "/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/GraphVisualization/"+FamilyNames[l]+"/"
                    AllString = []
                    # counter = 0
                    for files1 in os.listdir(directory1):
                        # counter += 1
                        # print(counter)

                        # print(files)
                        nodes_density = []
                        loc1 = directory1 + files1
                        g = ""
                        g2 = nx.drawing.nx_agraph.read_dot(loc1)
                        g2 = g2.to_directed()
                        g1 = nx.drawing.nx_agraph.read_dot(loc)
                        g1 = nx.DiGraph(g1)

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

                        # write_dot(g,"/home/ahmed/Documents/Projects/IoT_Attack_Journal/Code/SubGraphs/noComp.dot")
                        # exit()
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


                        maxShortestPath = np.max(shortestPathsArray)
                        minShortestPath = np.min(shortestPathsArray)
                        meanShortestPath = np.mean(shortestPathsArray)
                        medianShortestPath = np.median(shortestPathsArray)
                        stdShortestPath = np.std(shortestPathsArray)
                        closeness_list = list(closeness.values())
                        betweenness_list = list(shortest_betweenness.values())
                        degree_list = list(degree_centrality.values())
                        out = str(l1)+','+files+','+str(np.max(degree_list))+','+str(np.min(degree_list))+','+str(np.mean(degree_list))+','+str( np.median(degree_list))+','+str(np.std(degree_list))+','+str(np.max(betweenness_list))+','+str(np.min(betweenness_list))+','+str(np.mean(betweenness_list))+','+str( np.median(betweenness_list))+','+str(np.std(betweenness_list))+','+str(np.max(closeness_list))+','+str(np.min(closeness_list))+','+str(np.mean(closeness_list))+','+str( np.median(closeness_list))+','+str(np.std(closeness_list))+','+str(maxShortestPath)+','+str(minShortestPath)+','+str(meanShortestPath)+','+str(medianShortestPath)+','+str(stdShortestPath)+','+str(node_cnt)+','+str(edge_cnt)+','+str(density)+'\n'

                        z = [out.split(",")]
                        x = []
                        y = []
                        # for i in range(len(mainVisualizationVector)):
                        #     z.append(mainVisualizationVector[i].split(','))
                        for i in range(len(z)):
                            y.append(int(z[i][0]))
                            x.append(z[i][2:])
                        for i in range(len(x)):
                            for j in range(len(x[i])):
                                x[i][j]=  float(x[i][j])

                        x=np.asarray(x)
                        y=np.asarray(y)
                        for i in range(len(x)):
                            for j in range(len(x[i])):
                                if maxX[j] != 0 :
                                    x[i][j] = x[i][j]/maxX[j]

                        xAll = x.reshape((len(x),23,1))
                        predicting = model_argmax(sess, xprime, preds, xAll)
                        if predicting != y[0] :
                            if alreadyMisclass == 0:
                                Misclassification[predicting] += 1
                                alreadyMisclass = 1
                                # print("Misclassified")
                            if predicting == l:
                                TMisclassification[l] += 1
                                # print("Success")
                                break


        timeLast = time.time()  - timeFirst
        print(str(timeLast)+" seconds")
        print(All)
        print(Misclassification)
        print(TMisclassification)


def main(argv=None):
    mnist_tutorial_cw(viz_enabled=FLAGS.viz_enabled,
                      nb_epochs=FLAGS.nb_epochs,
                      batch_size=FLAGS.batch_size,
                      source_samples=FLAGS.source_samples,
                      learning_rate=FLAGS.learning_rate,
                      attack_iterations=FLAGS.attack_iterations,
                      model_path=FLAGS.model_path,
                      targeted=FLAGS.targeted)



if __name__ == '__main__':
    flags.DEFINE_boolean('viz_enabled', False, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 50, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 100, 'Size of training batches')
    flags.DEFINE_integer('source_samples', 1091, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', LEARNING_RATE,
                       'Learning rate for training')
    flags.DEFINE_string('model_path', os.path.join("models", "mnist"),
                        'Path to save or load the model file')
    flags.DEFINE_integer('attack_iterations', ATTACK_ITERATIONS,
                         'Number of iterations to run attack; 1000 is good')
    flags.DEFINE_boolean('targeted', False,
                         'Run the tutorial in targeted mode?')

    tf.app.run()
