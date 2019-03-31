from __future__ import absolute_import, division, print_function

import os
import numpy as np
import tensorflow as tf
from cell import *
from keras.datasets import cifar10
from keras.utils import to_categorical
import keras.backend as K
from keras.losses import categorical_crossentropy
from architect import *
from utils import *

tf.enable_eager_execution()  # enable eager execution process

from tensorflow.contrib import eager as tfe
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

tf.set_random_seed(10)
np.random.seed(10)

def main():

    # Construct keras model
    (x_train, y_train), (x_val, y_val) = parse_datasets(debugging_process=True)
    n_cells = 8
    num_classes = 10
    alphas_shape = [n_cells, n_cells, 8]
    n_filters = 16
    reduction_channels = False

    # Optimizing model weights and architecture weights iteratively.
    train_loss_results = []
    train_accuracy_results = []
    test_loss_results = []
    test_accuracy_results = []
    num_epochs = 60

    global_step = tf.Variable(0)
    device = '/gpu:0'   # calculating on gpu
    with tf.device(device):

        # Copy datasets on gpu
        batch_size = 64
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.batch(batch_size).shuffle(100)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(batch_size)

        model = Dart_cnn_model(num_classes, n_filters, reduction_channels)
        dummy_x = tf.zeros((1, 32, 32, 3))
        model.call(dummy_x)
        print(len(model.trainable_variables))
        architect = Architect(model)

        lr_alphas = 3e-4
        lr_weights = 0.001
        lr_xi = lr_weights
        alphas_optimizer = tf.train.AdamOptimizer(learning_rate=lr_alphas, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                                  use_locking=False, name='Adam_alphas')
        weights_optimizer = tf.train.AdamOptimizer(learning_rate=lr_weights, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                                   use_locking=False, name='Adam_weights')

        for epoch in range(num_epochs):

            train(model, train_dataset, val_dataset, architect, alphas_optimizer, weights_optimizer, lr_xi, epoch, global_step)
            validate(model, val_dataset, epoch, global_step)

            # Saving the model after each training epoch.
            # saver = tfe.Saver(model.trainable_variables)
            # saver.save('checkpoint.ckpt', global_step)


def parse_datasets(debugging_process):
    """
Args:
    debugging_process: if true, number of training samples is 1000, else 25000.
    """

    # Import and parse datasets
    subtract_pixel_mean = True
    num_classes = 10

    (x_train_, y_train_), (x_test, y_test) = cifar10.load_data()
    x_train_ = x_train_.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train_, axis=0)
        x_train_ -= x_train_mean
        x_test -= x_train_mean
    y_train_ = to_categorical(y_train_, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # We use 50% train samples as train datasets and 50 train samples as
    # validation samples, all test samples kept as test samples.
    x_train = x_train_[:25000]
    y_train = y_train_[:25000]
    x_val = x_train_[25000:]
    y_val = y_train_[25000:]

    if debugging_process:
        x_train = x_train[:1000]
        y_train = y_train[:1000]
        x_val = x_val[:1000]
        y_val = y_val[:1000]
        print("Script debugging process")
    else:
        print("Model training process.")

    print('x_train shape:', x_train.shape)
    print('x_val shape:', x_val.shape)

    return (x_train, y_train), (x_val, y_val)


def train(model, train_dataset, val_dataset, architect, alphas_optimizer, weights_optimizer, lr_xi, epoch, global_step):

    train_loss_avg = tfe.metrics.Mean()
    train_accuracy = tfe.metrics.Accuracy()
    var_alphas = [var for var in model.trainable_variables if 'arch_kernel' in var.name]
    var_weights = [var for var in model.trainable_variables if 'arch_kernel' not in var.name]
    for i, ((train_x, train_y), (val_x, val_y)) in enumerate(zip(train_dataset, val_dataset)):
        # Architecture optimization
        # if np.remainder(i, 3) == 0:
        grad_alphas = architect.unrolled_backward(train_x, train_y, val_x, val_y, lr_xi, weights_optimizer,
                                                  global_step)
        alphas_optimizer.apply_gradients(zip(grad_alphas, var_alphas),
                                         global_step)

        # Weights optimization
        grad_weights = grad(model, train_x, train_y, var_weights)
        grad_weights = [tf.clip_by_value(grad, -100, 100) for grad in grad_weights]
        weights_optimizer.apply_gradients(zip(grad_weights, var_weights),
                                          global_step)

        train_predictions, train_loss_value = loss(model, train_x, train_y)
        train_loss_avg(train_loss_value)
        train_accuracy(tf.argmax(train_predictions, axis=1, output_type=tf.int32),
                       tf.argmax(train_y, axis=1, output_type=tf.int32))
        print("{}th epoch, {}th iter, train loss:      {:0.3f}, accuracy: {:0.3%}.".format(epoch + 1,
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
    print(" {}th epoch, validation loss: {:0.3f}, accuracy: {:0.3%}.".format(epoch, val_loss_avg.result().numpy(),
                                                                                         val_accuracy.result().numpy()))
    print("___________________________________________________________________")


if __name__ == '__main__':
    main()

