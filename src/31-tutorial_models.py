from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools

import tensorflow as tf
from cleverhans.model import Model
from cleverhans.picklable_model import MLP, ReLU, Flatten, Linear
from cleverhans.picklable_model import Softmax
from keras.layers import Dense, Dropout, Activation, Flatten


class ModelBasicCNN(Model):

    def __init__(self, nb_classes=4,
                 nb_filters=46, dummy_input=tf.zeros((520, 23, 1))):
        Model.__init__(self, nb_classes=nb_classes)

        self.nb_filters = nb_filters
        self.nb_classes = nb_classes
        self.layer_names = ['input', 'conv_1', 'conv_2', 'maxpool_1', 'dropout_1', 'conv_3', 'conv_4', 'maxpool_2', 'dropout_2', 'flatten',
                             'logitDense','dropout_3','logits']
        self.layers = {}
        self.layer_acts = {}
        # layer definitions
        self.layers['conv_1'] = tf.layers.Conv1D(filters=self.nb_filters,
                                                 kernel_size=3,
                                                 padding='same',
                                                 activation=tf.nn.relu)
        self.layers['conv_2'] = tf.layers.Conv1D(filters=self.nb_filters,
                                                 kernel_size=3,
                                                 activation=tf.nn.relu)
        self.layers['maxpool_1'] = tf.layers.MaxPooling1D(pool_size=2, strides=2)
        self.layers['dropout_1'] = tf.layers.Dropout(rate=0.25)
        self.layers['conv_3'] = tf.layers.Conv1D(filters=2 * self.nb_filters,
                                                 kernel_size=3,
                                                 padding='same',
                                                 activation=tf.nn.relu)
        self.layers['conv_4'] = tf.layers.Conv1D(filters=2 * self.nb_filters,
                                                 kernel_size=3,
                                                 activation=tf.nn.relu)
        self.layers['maxpool_2'] = tf.layers.MaxPooling1D(pool_size=2, strides=2)
        self.layers['dropout_2'] = tf.layers.Dropout(rate=0.25)
        self.layers['flatten'] = tf.layers.Flatten()
        self.layers['logitDense'] = tf.layers.Dense(512,
                                                activation=tf.nn.relu)
        self.layers['dropout_3'] = tf.layers.Dropout(rate=0.5)
        self.layers['logits'] = tf.layers.Dense(self.nb_classes,
                                                activation=None)

        # Dummy fprop to activate the network.
        output = self.fprop(dummy_input)
    def fprop(self, x, **kwargs):
        del kwargs

        # Feed forward through the network layers
        for layer_name in self.layer_names:
            if layer_name == 'input':
                prev_layer_act = x
                continue
            else:
                self.layer_acts[layer_name] = self.layers[layer_name](
                                                                prev_layer_act)
                prev_layer_act = self.layer_acts[layer_name]
        # Adding softmax values to list of activations.
        self.layer_acts['probs'] = tf.nn.softmax(
                                        logits=self.layer_acts['logits'])
        return self.layer_acts

class ModelBasicDNN(Model):

    def __init__(self, nb_classes=4,
                 nb_filters=128, dummy_input=tf.zeros((9500, 23, 1))):
        Model.__init__(self, nb_classes=nb_classes)

        self.nb_filters = nb_filters
        self.nb_classes = nb_classes
        self.layer_names = ['input', 'conv_1', 'conv_2',  'dropout_1','flatten', 'logitDense_1','dropout_5','logitDense_2','dropout_6','logits']
        self.layers = {}
        self.layer_acts = {}
        # layer definitions
        self.layers['conv_1'] = tf.layers.Dense(100,
                                                activation=tf.nn.relu)

        self.layers['conv_2'] = tf.layers.Dense(100,
                                                activation=tf.nn.relu)
        self.layers['dropout_1'] = tf.layers.Dropout(rate=0.25)

        self.layers['flatten'] = tf.layers.Flatten()
        self.layers['logitDense_1'] = tf.layers.Dense(100,
                                                activation=tf.nn.relu)
        self.layers['dropout_5'] = tf.layers.Dropout(rate=0.25)

        self.layers['logitDense_2'] = tf.layers.Dense(100,
                                                activation=tf.nn.relu)
        self.layers['dropout_6'] = tf.layers.Dropout(rate=0.5)

        self.layers['logits'] = tf.layers.Dense(self.nb_classes,
                                                activation=None)

        # Dummy fprop to activate the network.
        output = self.fprop(dummy_input)
    def fprop(self, x, **kwargs):
        del kwargs

        # Feed forward through the network layers
        for layer_name in self.layer_names:
            if layer_name == 'input':
                prev_layer_act = x
                continue
            else:
                self.layer_acts[layer_name] = self.layers[layer_name](
                                                                prev_layer_act)
                prev_layer_act = self.layer_acts[layer_name]
        # Adding softmax values to list of activations.
        self.layer_acts['probs'] = tf.nn.softmax(
                                        logits=self.layer_acts['logits'])
        return self.layer_acts
