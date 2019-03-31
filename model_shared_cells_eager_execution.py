""" First only optimize weights for several epochs, then optimize archtecture and weights iteratively
"""

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import tensorflow as tf
from model_shared_cells import *
from keras.datasets import cifar10
from keras.utils import to_categorical
import keras.backend as K
from keras.losses import categorical_crossentropy
from architect import *
from utils import *
import copy

tf.enable_eager_execution()  # enable eager execution process

from tensorflow.contrib import eager as tfe
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

tf.set_random_seed(10)
np.random.seed(10)

def main(debugging_process, weight_arch_ratio):

    # Construct keras model
    (x_train, y_train), (x_val, y_val) = parse_datasets(debugging_process)
    n_cells = 8
    num_classes = 10
    alphas_shape = [n_cells, n_cells, 8]
    n_filters = 16
    reduction_channels = False

    train_loss_results = []
    train_accuracy_results = []
    test_loss_results = []
    test_accuracy_results = []
    num_epochs = 60

    global_step = tf.Variable(0)
    device = '/gpu:0'   # calculating on gpu
    print("_____________________________________________")
    print("Architecture search each {} epochs.".format(weight_arch_ratio))
    print("_____________________________________________")
    with tf.device(device):

        # Copy datasets on gpu
        batch_size = 64
        x_train_, y_train_ = copy.deepcopy((x_train, y_train))
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.batch(batch_size).shuffle(100)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(batch_size)

        model = DartSharedCells(num_classes, n_filters, reduction_channels)
        dummy_x = tf.zeros((1, 32, 32, 3))
        model.call(dummy_x)
        architect = Architect(model)

        lr_alphas = 0.0001
        lr_weights = 0.001
        lr_xi = lr_weights
        alphas_optimizer = tf.train.AdamOptimizer(learning_rate=lr_alphas, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                                  use_locking=False, name='Adam_alphas')
        weights_optimizer = tf.train.AdamOptimizer(learning_rate=lr_weights, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                                   use_locking=False, name='Adam_weights')

        for epoch in range(num_epochs):
            if np.remainder(epoch + 1, weight_arch_ratio) == 0:

                # Guarantee x_train_arch comes from x_train and changes it training epoch.
                x_train_arch, y_train_arch = datasets_shuffle_split(x_train_, y_train_, 0.25)
                print(x_train_arch.shape)
                train_arch_dataset = tf.data.Dataset.from_tensor_slices((x_train_arch, y_train_arch))
                train_arch_dataset = train_arch_dataset.batch(batch_size).shuffle(100)

                train_arch_weights(model, train_arch_dataset, val_dataset, architect, alphas_optimizer, weights_optimizer,
                                   lr_xi, epoch, global_step)
                validate(model, val_dataset, epoch, global_step)
            else:
                train_weights(model, train_dataset, weights_optimizer, epoch, global_step)
                validate(model, val_dataset, epoch, global_step)


def train_weights(model, train_dataset, weights_optimizer, epoch, global_step):

    train_loss_avg = tfe.metrics.Mean()
    train_accuracy = tfe.metrics.Accuracy()
    val_loss_avg = tfe.metrics.Mean()
    val_accuracy = tfe.metrics.Accuracy()

    var_alphas = [var for var in model.trainable_variables if 'arch_kernel' in var.name]
    var_weights = [var for var in model.trainable_variables if 'arch_kernel' not in var.name]
    for i, (train_x, train_y) in enumerate(train_dataset):

        # Weights optimization
        grad_weights = grad(model, train_x, train_y, var_weights)
        #grad_weights = [tf.clip_by_value(grad, -100, 100) for grad in grad_weights]
        weights_optimizer.apply_gradients(zip(grad_weights, var_weights),
                                          global_step)

        train_predictions, train_loss_value = loss(model, train_x, train_y)
        train_loss_avg(train_loss_value)
        train_accuracy(tf.argmax(train_predictions, axis=1, output_type=tf.int32),
                       tf.argmax(train_y, axis=1, output_type=tf.int32))
        print("{}th epoch, {}th iter, train loss:  {:0.3f}, accuracy: {:0.3%}.".format(epoch + 1,
                                                                                           i + 1,
                                                                                           train_loss_avg.result().numpy(),
                                                                                           train_accuracy.result().numpy()))

def train_arch_weights(model, train_dataset, val_dataset, architect, alphas_optimizer, weights_optimizer, lr_xi, epoch, global_step):

    train_loss_avg = tfe.metrics.Mean()
    train_accuracy = tfe.metrics.Accuracy()
    val_loss_avg = tfe.metrics.Mean()
    val_accuracy = tfe.metrics.Accuracy()

    var_alphas = [var for var in model.trainable_variables if 'arch_kernel' in var.name]
    var_weights = [var for var in model.trainable_variables if 'arch_kernel' not in var.name]
    for i, ((train_x, train_y), (val_x, val_y)) in enumerate(zip(train_dataset, val_dataset)):

        # We should call Architect after each time updates of model
        # Architecture optimization
        grad_alphas = architect.unrolled_backward(train_x, train_y, val_x, val_y, lr_xi, weights_optimizer,
                                                  global_step)
        alphas_optimizer.apply_gradients(zip(grad_alphas, var_alphas),
                                         global_step)

        val_prediction, val_loss_value = loss(model, val_x, val_y)
        val_loss_avg(val_loss_value)
        val_accuracy(tf.argmax(val_prediction, axis=1, output_type=tf.int32),
                     tf.argmax(val_y, axis=1, output_type=tf.int32))
        print("{}th epoch, {}th iter, valid loss:  {:0.3f}, accuracy: {:0.3%}.".format(epoch + 1,
                                                                                       i + 1,
                                                                                       val_loss_avg.result().numpy(),
                                                                                       val_accuracy.result().numpy()))
        # Weights optimization
        grad_weights = grad(model, train_x, train_y, var_weights)
        #grad_weights = [tf.clip_by_value(grad, -100, 100) for grad in grad_weights]
        weights_optimizer.apply_gradients(zip(grad_weights, var_weights),
                                          global_step)

        train_predictions, train_loss_value = loss(model, train_x, train_y)
        train_loss_avg(train_loss_value)
        train_accuracy(tf.argmax(train_predictions, axis=1, output_type=tf.int32),
                       tf.argmax(train_y, axis=1, output_type=tf.int32))
        print("{}th epoch, {}th iter, train loss:  {:0.3f}, accuracy: {:0.3%}.".format(epoch + 1,
                                                                                           i + 1,
                                                                                           train_loss_avg.result().numpy(),
                                                                                           train_accuracy.result().numpy()))


def validate(model, val_dataset, epoch, global_step):
    val_loss_avg = tfe.metrics.Mean()
    val_accuracy = tfe.metrics.Accuracy()
    for val_x, val_y in val_dataset:
        val_prediction, val_loss_value = loss(model, val_x, val_y)
        val_loss_avg(val_loss_value)
        val_accuracy(tf.argmax(val_prediction, axis=1, output_type=tf.int32),
                     tf.argmax(val_y, axis=1, output_type=tf.int32))
    print("___________________________________________________________________")
    print(" {}th epoch, validation loss: {:0.3f}, accuracy: {:0.3%}.".format(epoch+1, val_loss_avg.result().numpy(),
                                                                                         val_accuracy.result().numpy()))
    print("___________________________________________________________________")


if __name__ == '__main__':
    main(debugging_process=False, weight_arch_ratio=3)

