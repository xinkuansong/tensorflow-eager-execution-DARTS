from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
layers = tf.keras.layers
from utils import *

import os
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical
import keras.backend as K
from keras.losses import categorical_crossentropy

tf.enable_eager_execution()  # enable eager execution process

from tensorflow.contrib import eager as tfe
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

tf.set_random_seed(10)
np.random.seed(10)

class TestModel(tf.keras.Model):
    def __init__(self):
        super(TestModel, self).__init__()

        self.conv1 = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=2, padding='same', activation='relu',
                                   kernel_initializer='he_normal')
        self.conv2 = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                   kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.3)
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.3)
        self.bn3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(0.3)
        self.bn4 = layers.BatchNormalization()
        self.dropout4 = layers.Dropout(0.3)

        self.averagepooling = layers.AveragePooling2D(pool_size=(2))
        self.fl = layers.Flatten()
        self.dense = layers.Dense(10)

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.dropout3(x)

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.bn4(x)
        x = self.dropout4(x)

        x = self.averagepooling(x)
        x = self.fl(x)
        x = self.dense(x)

        return x

def loss(model, x, y):
    predictions = model(x)  # In the model, no softmax operation is performed
    loss = tf.losses.softmax_cross_entropy(logits=predictions, onehot_labels=y)  # softmax operations performed here
    return predictions, loss  # return predictions to avoid repeat calculation in calculating accuracy

def grad(model, x, y):
    with tf.GradientTape() as grad_tape:
        predictions, loss_value = loss(model, x, y)
        grads = grad_tape.gradient(loss_value, model.trainable_variables)
    return predictions, loss_value, grads

def main():

    debugging_process = True
    (x_train, y_train), (x_val, y_val) = parse_datasets(debugging_process)
    num_epochs = 60
    batch_size = 64

    train_loss_results = []
    train_accuracy_results = []
    global_step = tf.Variable(0)
    device = '/gpu:0'  # calculating on gpu
    with tf.device(device):

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.batch(batch_size).shuffle(100)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(batch_size)

        model = TestModel()
        dummy_x = tf.zeros((1, 32, 32, 3))
        model.call(dummy_x)

        lr = 1e-4
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                                  use_locking=False)

        for epoch in range(num_epochs):
            epoch_loss_avg = tfe.metrics.Mean()
            epoch_accuracy = tfe.metrics.Accuracy()
            test_loss = tfe.metrics.Mean()
            test_accuracy = tfe.metrics.Accuracy()
            temp = 0

            # In each epoch, optimizing model weights and fixing architecture weights.
            for x, y in train_dataset:
                temp += 1
                # print(model.trainable_variables)
                predictions, loss_value, grads = grad(model, x, y)
                optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                          global_step)
                epoch_loss_avg(loss_value)
                epoch_accuracy(tf.argmax(predictions, axis=1, output_type=tf.int32),
                               tf.argmax(y, axis=1, output_type=tf.int32))
                print("{}th epoch, {}th iter, loss: {:0.3f}, accuracy: {:0.3%}.".format(epoch + 1, temp,
                                                                                        epoch_loss_avg.result().numpy(),
                                                                                        epoch_accuracy.result().numpy()))
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())
            for x, y in val_dataset:
                predictions, loss_value = loss(model, x, y)
                test_loss(loss_value)
                test_accuracy(tf.argmax(predictions, axis=1, output_type=tf.int32),
                              tf.argmax(y, axis=1, output_type=tf.int32))
            print(
                "______________________________________________________________________________________________________________")
            print(
                "Epoch {}th, Validation loss: {:0.3f}, accuracy: {:0.3%}.".format(epoch + 1, test_loss.result().numpy(),
                                                                                  test_accuracy.result().numpy()))
            print(
                "______________________________________________________________________________________________________________")




if __name__ == '__main__':
    main()

































































































