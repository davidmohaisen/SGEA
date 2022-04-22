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
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import time
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
from cleverhans_tutorials.tutorial_models import ModelBasicDNN
from six.moves import xrange
from sklearn.model_selection import train_test_split
import keras
from cleverhans.attacks import SPSA
from cleverhans.attacks import SaliencyMapMethod
FLAGS = flags.FLAGS

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
    """
    MNIST tutorial for Carlini and Wagner's attack
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param viz_enabled: (boolean) activate plots of adversarial examples
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param nb_classes: number of output classes
    :param source_samples: number of test inputs to attack
    :param learning_rate: learning rate for training
    :param model_path: path to the model file
    :param targeted: should we run a targeted attack? or untargeted?
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get MNIST test data
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
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    nb_filters = 46

    # Define TF model graph
    model = ModelBasicDNN(nb_classes, nb_filters)
    preds = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=0.1)
    print("Defined TensorFlow model graph.")

    saver = tf.train.Saver()
    saver.restore(sess, "/home/ahmed/Documents/Projects/IoT_Attack_Journal/Models/TrainedModelDNN.ckpt")

    print("Model restored.")

    xEval = []
    yEval = []
    predicting = model_argmax(sess, x, preds, x_test)
    for i in range(len(x_test)):
        if y_test[i][predicting[i]] == 1 :
            xEval.append(x_test[i])
            yEval.append(y_test[i])
    xEval = np.asarray(xEval)
    yEval = np.asarray(yEval)

    source_samples = len(xEval)
    print(len(xEval))
    nb_adv_per_sample = str(nb_classes - 1) if targeted else '1'
    print('Crafting ' + str(source_samples) + ' * ' + nb_adv_per_sample +
          ' adversarial examples')
    print("This could take some time ...")

    # Instantiate a CW attack object

    if targeted:
        if viz_enabled:
            # Initialize our array for grid visualization
            grid_shape = (nb_classes, nb_classes, img_rows, img_cols,
                          nchannels)
            grid_viz_data = np.zeros(grid_shape, dtype='f')

            adv_inputs = np.array(
                [[instance] * nb_classes for instance in xEval[idxs]],
                dtype=np.float32)
        else:
            adv_inputs = np.array(
                [[instance] * nb_classes for
                 instance in xEval[:source_samples]], dtype=np.float32)

        one_hot = np.zeros((nb_classes, nb_classes))
        one_hot[np.arange(nb_classes), np.arange(nb_classes)] = 1

        adv_inputs = adv_inputs.reshape(
            (source_samples * nb_classes, img_rows, img_cols, nchannels))
        adv_ys = np.array([one_hot] * source_samples,
                          dtype=np.float32).reshape((source_samples *
                                                     nb_classes, nb_classes))
        yname = "y_target"
    else:
        adv_inputs = xEval[:source_samples]

        adv_ys = None
        yname = "y"

    timeFirst = time.time()
##########################################
# CW
###############################################
    # cw = CarliniWagnerL2(model, back='tf', sess=sess)
    # methodName = "CarliniWagnerL2"
    # cw_params = {
    #         'max_iterations':200,
    #         'learning_rate':0.1,
    #         'clip_min': 0,
    #         'batch_size': len(adv_inputs[:300]),
    #         'clip_max': 1
    #              }
    # advA = cw.generate_np(adv_inputs[:300],
    #                      **cw_params)
    # cw_params = {
    #         'max_iterations':200,
    #         'learning_rate':0.1,
    #         'clip_min': 0,
    #         'batch_size': len(adv_inputs[300:600]),
    #         'clip_max': 1
    #              }
    # advB = cw.generate_np(adv_inputs[300:600],
    #                      **cw_params)
    # cw_params = {
    #         'max_iterations':200,
    #         'learning_rate':0.1,
    #         'clip_min': 0,
    #         'batch_size': len(adv_inputs[600:900]),
    #         'clip_max': 1
    #              }
    # adv3 = cw.generate_np(adv_inputs[600:900],
    #                      **cw_params)
    # cw_params = {
    #         'max_iterations':200,
    #         'learning_rate':0.1,
    #         'clip_min': 0,
    #         'batch_size': len(adv_inputs[900:1200]),
    #         'clip_max': 1
    #              }
    # adv4 = cw.generate_np(adv_inputs[900:1200],
    #                      **cw_params)
    # cw_params = {
    #         'max_iterations':200,
    #         'learning_rate':0.1,
    #         'clip_min': 0,
    #         'batch_size': len(adv_inputs[1200:1500]),
    #         'clip_max': 1
    #              }
    # adv5 = cw.generate_np(adv_inputs[1200:1500],
    #                      **cw_params)
    # cw_params = {
    #         'max_iterations':200,
    #         'learning_rate':0.1,
    #         'clip_min': 0,
    #         'batch_size': len(adv_inputs[1500:]),
    #         'clip_max': 1
    #              }
    # adv6 = cw.generate_np(adv_inputs[1500:],
    #                      **cw_params)
    # adv1 = np.concatenate((advA, advB,adv3), axis=0)
    # adv2 = np.concatenate((adv4, adv5, adv6), axis=0)
    # advLabels1 = model_argmax(sess, x, preds, adv1)
    # advLabels1= keras.utils.to_categorical(advLabels1, 4)
    # advLabels2 = model_argmax(sess, x, preds, adv2)
    # advLabels2= keras.utils.to_categorical(advLabels2, 4)


# ###########################################
# # DeepFool
# ################################################
    # DeepFool_params = {
    #         'max_iter': 100,
    #         'nb_candidate': 2,
    #         'clip_min': 0
    #              }
    # df = DeepFool(model, back='tf', sess=sess)
    # methodName = "DeepFool"
    #
    # advA = df.generate_np(adv_inputs[:300],
    #                      **DeepFool_params)
    # advB = df.generate_np(adv_inputs[300:600],
    #                      **DeepFool_params)
    # adv3 = df.generate_np(adv_inputs[600:900],
    #                      **DeepFool_params)
    # adv4 = df.generate_np(adv_inputs[900:1200],
    #                      **DeepFool_params)
    # adv5 = df.generate_np(adv_inputs[1200:1500],
    #                      **DeepFool_params)
    # adv6 = df.generate_np(adv_inputs[1500:],
    #                      **DeepFool_params)
    # adv1 = np.concatenate((advA, advB,adv3), axis=0)
    # adv2 = np.concatenate((adv4, adv5, adv6), axis=0)
    # advLabels1 = model_argmax(sess, x, preds, adv1)
    # advLabels1= keras.utils.to_categorical(advLabels1, 4)
    # advLabels2 = model_argmax(sess, x, preds, adv2)
    # advLabels2= keras.utils.to_categorical(advLabels2, 4)
# ###########################################
# # FastGradientMethod
################################################
    # FGSM_params = {
    #         'eps': 0.3,
    #         'clip_min': 0,
    #         'clip_max':1
    #              }
    # FGSM = FastGradientMethod(model, back='tf', sess=sess)
    # methodName = "FGSM"
    #
    # advA = FGSM.generate_np(adv_inputs[:300],
    #                      **FGSM_params)
    # advB = FGSM.generate_np(adv_inputs[300:600],
    #                      **FGSM_params)
    # adv3 = FGSM.generate_np(adv_inputs[600:900],
    #                      **FGSM_params)
    # adv4 = FGSM.generate_np(adv_inputs[900:1200],
    #                      **FGSM_params)
    # adv5 = FGSM.generate_np(adv_inputs[1200:1500],
    #                      **FGSM_params)
    # adv6 = FGSM.generate_np(adv_inputs[1500:],
    #                      **FGSM_params)
    # adv1 = np.concatenate((advA, advB,adv3), axis=0)
    # adv2 = np.concatenate((adv4, adv5, adv6), axis=0)
    # advLabels1 = model_argmax(sess, x, preds, adv1)
    # advLabels1= keras.utils.to_categorical(advLabels1, 4)
    # advLabels2 = model_argmax(sess, x, preds, adv2)
    # advLabels2= keras.utils.to_categorical(advLabels2, 4)
    #

# ###########################################
# # MomentumIterativeMethod Default
# ################################################
    # MIM_params = {
    #         'eps': 0.3,
    #         'nb_iter': 10,
    #         'eps_iter': .06,
    #         'clip_min': 0
    #              }
    # MIM = MomentumIterativeMethod(model, back='tf', sess=sess)
    # methodName = "MomentumIterativeMethod"
    #
    # advA = MIM.generate_np(adv_inputs[:300],
    #                      **MIM_params)
    # advB = MIM.generate_np(adv_inputs[300:600],
    #                      **MIM_params)
    # adv3 = MIM.generate_np(adv_inputs[600:900],
    #                      **MIM_params)
    # adv4 = MIM.generate_np(adv_inputs[900:1200],
    #                      **MIM_params)
    # adv5 = MIM.generate_np(adv_inputs[1200:1500],
    #                      **MIM_params)
    # adv6 = MIM.generate_np(adv_inputs[1500:],
    #                      **MIM_params)
    # adv1 = np.concatenate((advA, advB,adv3), axis=0)
    # adv2 = np.concatenate((adv4, adv5, adv6), axis=0)
    # advLabels1 = model_argmax(sess, x, preds, adv1)
    # advLabels1= keras.utils.to_categorical(advLabels1, 4)
    # advLabels2 = model_argmax(sess, x, preds, adv2)
    # advLabels2= keras.utils.to_categorical(advLabels2, 4)

# ###########################################
# # MadryEtAl Default
# ################################################
    # Madry_params = {
    #         'eps': 0.3,
    #         'nb_iter': 40,
    #         'clip_min': 0,
    #         'clip_max': 1
    #          }
    # Madry = MadryEtAl(model, back='tf', sess=sess)
    # methodName = "MadryEtAlDefault"
    #
    # advA = Madry.generate_np(adv_inputs[:300],
    #                      **Madry_params)
    # advB = Madry.generate_np(adv_inputs[300:600],
    #                      **Madry_params)
    # adv3 = Madry.generate_np(adv_inputs[600:900],
    #                      **Madry_params)
    # adv4 = Madry.generate_np(adv_inputs[900:1200],
    #                      **Madry_params)
    # adv5 = Madry.generate_np(adv_inputs[1200:1500],
    #                      **Madry_params)
    # adv6 = Madry.generate_np(adv_inputs[1500:],
    #                      **Madry_params)
    # adv1 = np.concatenate((advA, advB,adv3), axis=0)
    # adv2 = np.concatenate((adv4, adv5, adv6), axis=0)
    # advLabels1 = model_argmax(sess, x, preds, adv1)
    # advLabels1= keras.utils.to_categorical(advLabels1, 4)
    # advLabels2 = model_argmax(sess, x, preds, adv2)
    # advLabels2= keras.utils.to_categorical(advLabels2, 4)
# ###########################################
# # VirtualAdversarialMethod Default
# ################################################
    # VAM_params = {
    #         'eps': 0.3,
    #         'clip_min': 0,
    #         'num_iterations':40,
    #         'clip_max':1
    #              }
    # VAM = VirtualAdversarialMethod(model, back='tf', sess=sess)
    # methodName = "VirtualAdversarialMethodDefault"
    #
    # advA = VAM.generate_np(adv_inputs[:300],
    #                      **VAM_params)
    # advB = VAM.generate_np(adv_inputs[300:600],
    #                      **VAM_params)
    # adv3 = VAM.generate_np(adv_inputs[600:900],
    #                      **VAM_params)
    # adv4 = VAM.generate_np(adv_inputs[900:1200],
    #                      **VAM_params)
    # adv5 = VAM.generate_np(adv_inputs[1200:1500],
    #                      **VAM_params)
    # adv6 = VAM.generate_np(adv_inputs[1500:],
    #                      **VAM_params)
    # adv1 = np.concatenate((advA, advB,adv3), axis=0)
    # adv2 = np.concatenate((adv4, adv5, adv6), axis=0)
    # advLabels1 = model_argmax(sess, x, preds, adv1)
    # advLabels1= keras.utils.to_categorical(advLabels1, 4)
    # advLabels2 = model_argmax(sess, x, preds, adv2)
    # advLabels2= keras.utils.to_categorical(advLabels2, 4)
# ###########################################
# # ElasticNet Method
# ################################################
    # ElasticNetMethod_params = {
    #         'max_iterations':250,
    #         'batch_size' : 100,
    #         'learning_rate' : 0.1,
    #         'binary_search_steps' : 5
    #
    # }
    # Elastic = ElasticNetMethod(model, back='tf', sess=sess)
    # methodName = "ElasticNet"
    # advA = Elastic.generate_np(adv_inputs[:300],
    #                      **ElasticNetMethod_params)
    # advB = Elastic.generate_np(adv_inputs[300:600],
    #                      **ElasticNetMethod_params)
    # adv3 = Elastic.generate_np(adv_inputs[600:900],
    #                      **ElasticNetMethod_params)
    # adv4 = Elastic.generate_np(adv_inputs[900:1200],
    #                      **ElasticNetMethod_params)
    # adv5 = Elastic.generate_np(adv_inputs[1200:1500],
    #                      **ElasticNetMethod_params)
    # ElasticNetMethod_params = {
    #         'max_iterations':250,
    #         'batch_size' : len(adv_inputs[1500:]),
    #         'learning_rate' : 0.1,
    #         'binary_search_steps' : 5
    # }
    # adv6 = Elastic.generate_np(adv_inputs[1500:],
    #                      **ElasticNetMethod_params)
    #
    # adv1 = np.concatenate((advA, advB,adv3), axis=0)
    # adv2 = np.concatenate((adv4, adv5, adv6), axis=0)
    # advLabels1 = model_argmax(sess, x, preds, adv1)
    # advLabels1= keras.utils.to_categorical(advLabels1, 4)
    # advLabels2 = model_argmax(sess, x, preds, adv2)
    # advLabels2= keras.utils.to_categorical(advLabels2, 4)
# ###########################################
# # jsma
# ################################################
    # SMM_params = {
    #         'theta': 0.3,
    #         'gamma': 0.6,
    #         'clip_min': 0
    #     }
    # SMM = SaliencyMapMethod(model, back='tf', sess=sess)
    # methodName = "JSMA"
    #
    # advA = SMM.generate_np(adv_inputs[:300],
    #                      **SMM_params)
    # advB = SMM.generate_np(adv_inputs[300:600],
    #                      **SMM_params)
    # adv3 = SMM.generate_np(adv_inputs[600:900],
    #                      **SMM_params)
    # adv4 = SMM.generate_np(adv_inputs[900:1200],
    #                      **SMM_params)
    # adv5 = SMM.generate_np(adv_inputs[1200:1500],
    #                      **SMM_params)
    # adv6 = SMM.generate_np(adv_inputs[1500:],
    #                      **SMM_params)
    # adv1 = np.concatenate((advA, advB,adv3), axis=0)
    # adv2 = np.concatenate((adv4, adv5, adv6), axis=0)
    # advLabels1 = model_argmax(sess, x, preds, adv1)
    # advLabels1= keras.utils.to_categorical(advLabels1, 4)
    # advLabels2 = model_argmax(sess, x, preds, adv2)
    # advLabels2= keras.utils.to_categorical(advLabels2, 4)
    #



    timeLast = time.time()  - timeFirst
    print(str(timeLast)+" seconds")
    print(str(len(adv1)+len(adv2)))

    changed = 0
    counter = 0
    changingTimes = [0] * 23
    for i in range(len(adv1)):
        for j in range(len(adv1[i])):
            if int(adv1[i][j][0]*100)  != int(adv_inputs[counter][j][0]*100) :
                changed += 1
                changingTimes[j] += 1
        counter += 1
    for i in range(len(adv2)):
        for j in range(len(adv2[i])):
            if int(adv2[i][j][0]*100)  != int(adv_inputs[counter][j][0]*100) :
                changed += 1
                changingTimes[j] += 1
        counter += 1
    adv = np.concatenate((adv1, adv2), axis=0)
    advLabels = np.concatenate((advLabels1, advLabels2), axis=0)




    eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}
    if targeted:
        adv_accuracy = model_eval(
            sess, x, y, preds, adv, adv_ys, args=eval_params)
    else:
        adv_accuracy = 1 - \
            model_eval(sess, x, y, preds, adv, yEval[
                         :source_samples], args=eval_params)
    print('Avg. rate of successful adv. examples {0:.4f}'.format((adv_accuracy)))
    print(changingTimes)
    print("Changed: "+str(changed/len(adv_inputs)))

    exit()
    ############ Save Adv ############
    print('Files Saving...')

    # outputVector= open('/home/ahmed/Documents/sdn/CNNAdversary/'+methodName+'Images.txt', 'w')
    outputVectorOrg= open('/home/ahmed/Documents/Projects/IoT_Attack_Journal/OffShelfAttack/'+methodName+'ImagesOriginalCNN.txt', 'w')
    for i in range(len(x_test)):
        np.savetxt(outputVectorOrg, x_test[i].reshape(23), fmt='%-7.2f')
        outputVectorOrg.write('# End OF Sample\n')

    outputVector= open('/home/ahmed/Documents/Projects/IoT_Attack_Journal/OffShelfAttack/'+methodName+'ImagesCNN.txt', 'w')
    for i in range(len(adv1)):
        np.savetxt(outputVector, adv1[i].reshape(23), fmt='%-7.2f')
        outputVector.write('# End OF Sample\n')
    for i in range(len(adv2)):
        np.savetxt(outputVector, adv2[i].reshape(23), fmt='%-7.2f')
        outputVector.write('# End OF Sample\n')
    # outputLabels= open('/home/ahmed/Documents/sdn/CNNAdversary/'+methodName+'CorrectLabels.txt', 'w')
    outputLabels= open('/home/ahmed/Documents/Projects/IoT_Attack_Journal/OffShelfAttack/'+methodName+'CorrectLabelsCNN.txt', 'w')
    outputLabels.write('# Array shape: 34717 , 68 , 1 \n')
    np.savetxt(outputLabels, yEval, fmt='%-7.2f')
    # outputADVLabels= open('/home/ahmed/Documents/sdn/CNNAdversary/'+methodName+'AdversaryLabels.txt', 'w')
    outputADVLabels= open('/home/ahmed/Documents/Projects/IoT_Attack_Journal/OffShelfAttack/'+methodName+'AdversaryLabelsCNN.txt', 'w')
    np.savetxt(outputADVLabels, advLabels1, fmt='%-7.2f')
    np.savetxt(outputADVLabels, advLabels2, fmt='%-7.2f')
    outputLabels.close()
    outputVector.close()
    outputADVLabels.close()
    print('Files Saved')


    ############ Save Adv ############
    adv = np.concatenate((adv1, adv2), axis=0)
    advLabels = np.concatenate((advLabels1, advLabels2), axis=0)


    eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}
    if targeted:
        adv_accuracy = model_eval(
            sess, x, y, preds, adv, adv_ys, args=eval_params)
    else:
        adv_accuracy = 1 - \
            model_eval(sess, x, y, preds, adv, yEval[
                         :source_samples], args=eval_params)



    print('Avg. rate of successful adv. examples {0:.4f}'.format((adv_accuracy)))
    report.clean_train_adv_eval = 1. - ((adv_accuracy))
    adv
    yEval
    M2B = 0
    Mtotal = 0
    Btotal = 0
    B2M = 0
    for i in range(len(advLabels)):
        if int(yEval[i][0]) == 1 :
            Btotal += 1
            if int(advLabels[i][0]) == 0 :
                B2M += 1
        if int(yEval[i][1]) == 1 :
            Mtotal += 1
            if int(advLabels[i][1]) == 0 :
                M2B += 1
    print("Mtotal = "+str(Mtotal))
    print("Btotal = "+str(Btotal))
    print("M2B = " +str(M2B))
    print("B2M = " +str(B2M))
    print("M2BR = " +str(M2B/Mtotal))
    print("B2MR = " +str(B2M/Btotal))



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
