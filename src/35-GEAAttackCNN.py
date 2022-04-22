
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

import logging
import os
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import DeepFool
from cleverhans.attacks import MadryEtAl
from cleverhans.loss import CrossEntropy
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import VirtualAdversarialMethod
from cleverhans.attacks import ElasticNetMethod
from cleverhans.utils import grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import train, model_eval, tf_model_load, model_argmax
from cleverhans_tutorials.tutorial_models import ModelBasicCNN
from six.moves import xrange
from sklearn.model_selection import train_test_split
import keras
from cleverhans.attacks import SPSA
from cleverhans.attacks import SaliencyMapMethod
FLAGS = flags.FLAGS
import networkx as nx
import os
import sys
import pygraphviz
import numpy as np
import time
import keras
import pickle
LEARNING_RATE = .001
CW_LEARNING_RATE = .2
ATTACK_ITERATIONS = 1000
def mnist_tutorial_cw(train_start=0, train_end=60000, test_start=0,
                      test_end=10000, viz_enabled=True, nb_epochs=6,
                      batch_size=128, source_samples=10,
                      learning_rate=LEARNING_RATE,
                      attack_iterations=ATTACK_ITERATIONS,
                      model_path=os.path.join("models", "mnist"),
                      targeted=True):

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()



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
    xprime = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    nb_filters = 46

    # Define TF model graph
    model = ModelBasicCNN(nb_classes, nb_filters)
    preds = model.get_logits(xprime)
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

    train(sess, loss, xprime, y, x_train, y_train, args=train_params,
          save=os.path.exists("models"), rng=rng)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, xprime, y, preds, x_test, y_test, args=eval_params)
    print('Test accuracy on clean test examples: {0}'.format(accuracy))


    base = "/home/ahmed/Documents/Projects/IoT_Attack_Journal/GEA/TestList/"
    saveBase = "/home/ahmed/Documents/Projects/IoT_Attack_Journal/GEA/Features/"
    families = ["Benign/","Gafgyt/","Mirai/","Tsunami/"]
    classes = ["Benign","Gafgyt","Mirai","Tsunami"]
    size = ["Minimum","Median","Maximum"]


    for l1 in range(len(classes)):
        for l2 in range(len(size)):
            CSV = saveBase+families[l1]+size[l2]+'/features.csv'
            print("______________________________")
            print(classes[l1],size[l2])

            mainVisualizationVector = open(CSV)
            mainVisualizationVector = mainVisualizationVector.readlines()
            org = open('/home/ahmed/Documents/Projects/IoT_Attack_Journal/DS/FeatureVector/Features.csv')
            org = org.readlines()

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

            for i in range(len(x)):
                for j in range(len(x[i])):
                    if maxX[j] != 0 :
                        x[i][j] = x[i][j]/maxX[j]

            xAll = x.reshape((len(x),23,1))
            y_old = y
            # yAll = keras.utils.to_categorical(y, 4)

            predicting = model_argmax(sess, xprime, preds, xAll)
            print(len(predicting))
            # print(predicting[:10])
            MC = 0
            TMC = 0
            for c in range(len(predicting)):
                if y[c] != predicting[c]:
                    MC += 1
                    if predicting[c] == l1:
                        TMC += 1
            print("Misclassification Rate: ",MC/len(predicting))
            print("Targetted Misclassification Rate: ",TMC/len(predicting))



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
    flags.DEFINE_integer('nb_epochs', 100, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('source_samples', 34717, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', LEARNING_RATE,
                       'Learning rate for training')
    flags.DEFINE_string('model_path', os.path.join("models", "mnist"),
                        'Path to save or load the model file')
    flags.DEFINE_integer('attack_iterations', ATTACK_ITERATIONS,
                         'Number of iterations to run attack; 1000 is good')
    flags.DEFINE_boolean('targeted', False,
                         'Run the tutorial in targeted mode?')

    tf.app.run()
