"""
This tutorial shows how to generate adversarial examples
using C&W attack in white-box setting.
The original paper can be found at:
https://nicholas.carlini.com/papers/2017_sp_nnrobustattacks.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

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
from cleverhans_tutorials.tutorial_models import ModelBasicCNN
from six.moves import xrange
from sklearn.model_selection import train_test_split,KFold
import keras
from cleverhans.attacks import SPSA
from cleverhans.attacks import SaliencyMapMethod
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

    y_trainN = []
    for i in range(len(y_train)):
        value = np.argmax(y_train[i])
        if value != 0 :
            value = 1
        y_trainN.append(value)
    y_testN = []
    for i in range(len(y_test)):
        value = np.argmax(y_test[i])
        if value != 0 :
            value = 1
        y_testN.append(value)
    y_train = y_trainN
    y_test = y_testN
    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)

    img_rows, img_cols = x_train.shape[1:3]
    nb_classes = y_train.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    nb_filters = 46

    # Define TF model graph
    model = ModelBasicCNN(nb_classes, nb_filters)
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
    saver = tf.train.Saver()
    save_path = saver.save(sess, "/home/ahmed/Documents/Projects/IoT_Attack_Journal/Models/detection/CNN/TrainedModelCNN.ckpt")
    print("Model saved in path: %s" % save_path)






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
