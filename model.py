""" Creating a CNN model which can be used in eager execution.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
layers = tf.keras.layers
from utils import *

class DartCnnModel(tf.keras.Model):

    """ Buidling a CNN model based on DART
    Args:
        input_shape: input shape of the model;
        n_cells: number of CNN cell.

        return: a CNN model
    Specifics:
        (1) cell 3 and cell 6 are reduction cell, feature size divide two, number of filters multiply two
        (2) inputs of cell 4 and cell 7 should be [cell_3_outputs, cell_3_outputs] and [cell_6_outputs, cell_6_outputs]
    """

    def __init__(self,
                 num_classes,
                 n_filters,
                 reduction_channels):
        super(DartCnnModel, self).__init__()

        self.num_classes = num_classes

        self.n_filters = n_filters
        self.reduction_channels = reduction_channels

        self.cell_1 = Cell(True, False, self.n_filters, self.reduction_channels)
        self.cell_2 = Cell(True, False, self.n_filters, self.reduction_channels)

        #self.n_filters *= 2
        self.cell_3 = Cell(False, True, self.n_filters * 2, self.reduction_channels)
        self.cell_4 = Cell(False, False, self.n_filters * 2, self.reduction_channels)
        self.cell_5 = Cell(False, False, self.n_filters * 2, self.reduction_channels)

        #self.n_filters *= 2
        self.cell_6 = Cell(False, True, self.n_filters * 4, self.reduction_channels)
        self.cell_7 = Cell(False, False, self.n_filters * 4, self.reduction_channels)
        self.cell_8 = Cell(False, False, self.n_filters * 4, self.reduction_channels)

        self.averagepooling_1 = layers.AveragePooling2D(pool_size=(8, 8))
        self.fl_1 = layers.Flatten()
        self.dense_1 = layers.Dense(self.num_classes)


    def call(self, inputs):

        input_1 = input_2 = inputs
        cell_1_inputs = [input_1, input_2]
        cell_1_outputs = self.cell_1(cell_1_inputs)
        cell_2_inputs = [cell_1_inputs[1], cell_1_outputs]
        cell_2_outputs = self.cell_2(cell_2_inputs)

        cell_3_inputs = [cell_2_inputs[1], cell_2_outputs]
        cell_3_outputs = self.cell_3(cell_3_inputs)
        cell_4_inputs = [cell_3_outputs, cell_3_outputs]
        cell_4_outputs = self.cell_4(cell_4_inputs)
        cell_5_inputs = [cell_4_inputs[1], cell_4_outputs]
        cell_5_outputs = self.cell_5(cell_5_inputs)

        cell_6_inputs = [cell_5_inputs[1], cell_5_outputs]
        cell_6_outputs = self.cell_6(cell_6_inputs)
        cell_7_inputs = [cell_6_outputs, cell_6_outputs]
        cell_7_outputs = self.cell_7(cell_7_inputs)
        cell_8_inputs = [cell_7_inputs[1], cell_7_outputs]
        cell_8_outputs = self.cell_8(cell_8_inputs)

        x = self.averagepooling_1(cell_8_outputs)
        x = self.fl_1(x)
        outputs = self.dense_1(x)

        return outputs


class Cell(tf.keras.Model):
    """ Calculate the output of a cell unit
    Args:
        inputs: outputs from former two cells;
        alphas: encodings of architectures, shape should be n_nodes * n_nodes * n_ops;
        ops: optional operations in each edge;

        reduction_flag: if the cell is reduction or not;
        n_filters: number of filters in each operation;
        reduction_channels: ratio of channels number reduction.

        return: cell output
    Specifics:
        (1) first two nodes are outputs of former two cells;
        (2) all intermediate nodes(not including the output node) are connected with first two nodes,
            strides of those edges are determined by reduction_stride;
        (3) strides of all edges between intermediate cells are deterimined by stride;
        (4) all intermediate nodes are connected with output node and edge strides determined by stride.

    """
    def __init__(self,
                 first_two_cell,
                 reduction_flag,
                 n_filters,
                 reduction_channels):
        super(Cell, self).__init__()
        self.first_two_cell = first_two_cell
        self.reduction_flag = reduction_flag
        self.n_filters = n_filters
        self.reduction_channels = reduction_channels
        self.stride = 1

        if (self.reduction_flag) and (not self.first_two_cell):
            self.reduction_stride = 2
        elif (not self.reduction_flag) and (self.first_two_cell):
            self.reduction_stride = 1000
        else:
            self.reduction_stride = 1

        self.edge_13 = Node(self.n_filters, self.reduction_stride)
        self.edge_23 = Node(self.n_filters, self.reduction_stride)

        self.edge_14 = Node(self.n_filters, self.reduction_stride)
        self.edge_24 = Node(self.n_filters, self.reduction_stride)
        self.edge_34 = Node(self.n_filters, self.stride)

        self.edge_15 = Node(self.n_filters, self.reduction_stride)
        self.edge_25 = Node(self.n_filters, self.reduction_stride)
        self.edge_35 = Node(self.n_filters, self.stride)
        self.edge_45 = Node(self.n_filters, self.stride)

        self.edge_16 = Node(self.n_filters, self.reduction_stride)
        self.edge_26 = Node(self.n_filters, self.reduction_stride)
        self.edge_36 = Node(self.n_filters, self.stride)
        self.edge_46 = Node(self.n_filters, self.stride)
        self.edge_56 = Node(self.n_filters, self.stride)

        self.output_edge_3 = Node(self.n_filters, self.stride)
        self.output_edge_4 = Node(self.n_filters, self.stride)
        self.output_edge_5 = Node(self.n_filters, self.stride)
        self.output_edge_6 = Node(self.n_filters, self.stride)

        self.conv1 = layers.Conv2D(filters=self.n_filters * self.reduction_channels, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal')

    def call(self, inputs, training=True):

        input_1, input_2 = inputs
        node_3 = layers.add([self.edge_13(input_1),
                             self.edge_23(input_2)])
        node_4 = layers.add([self.edge_14(input_1),
                             self.edge_24(input_2),
                             self.edge_34(node_3)])
        node_5 = layers.add([self.edge_15(input_1),
                             self.edge_25(input_2),
                             self.edge_35(node_3),
                             self.edge_45(node_4)])
        node_6 = layers.add([self.edge_16(input_1),
                             self.edge_26(input_2),
                             self.edge_36(node_3),
                             self.edge_46(node_4),
                             self.edge_56(node_5)])
        cell_outputs = layers.add([self.output_edge_3(node_3),
                                   self.output_edge_4(node_4),
                                   self.output_edge_5(node_5),
                                   self.output_edge_6(node_6)])
        if self.reduction_channels:
            return self.conv1(cell_outputs)

        return cell_outputs


class Node(tf.keras.Model):
    """ Calculate the output based on output of former node.
    Args:
        intermediate_input: output from former node;
        filters: number of filters
        stride: stride used in operations;

        alpha_node: weights used for each operations
        ops: 8 optional operations
             3 * 3 Conv2D
             5 * 5 Conv2D
             3 * 3 SeparableConv2D
             5 * 5 SeparableConv2D
             3 * 3 MaxPooling2D
             3 * 3 AveragePooling2D
             identity, work as residual connection
             zero, work as disconnection

    specifics: the node class contains two different situation
             (1) for first two cells, no maxpooling, averagepooling, identity, and zero;
             (2) for reduction cells, no maxpooling, averagepooling, identity, and zero(because number of filters will doubled);
             (3) for cells after reduction cells, no maxpooling, averagepooling, identity, and zero;

    """
    def __init__(self,
                 n_filters,
                 reduction_stride):
        super(Node, self).__init__()

        self.reduction_stride = reduction_stride
        self.n_filters = n_filters
        self.DROPOUT = 0.4

        if self.reduction_stride == 1:

            self.conv1 = layers.Conv2D(filters=self.n_filters, kernel_size=(3, 3), strides=self.reduction_stride, padding='same', activation='relu',
                                       kernel_initializer='he_normal')
            self.bn1 = layers.BatchNormalization()
            self.dropout1 = layers.Dropout(self.DROPOUT)

            self.conv2 = layers.Conv2D(filters=self.n_filters, kernel_size=(5, 5), strides=self.reduction_stride, padding='same',
                                       activation='relu',
                                       kernel_initializer='he_normal')
            self.bn2 = layers.BatchNormalization()
            self.dropout2 = layers.Dropout(self.DROPOUT)

            self.separable_conv3 = layers.SeparableConv2D(filters=self.n_filters, kernel_size=(3, 3), strides=self.reduction_stride, padding='same',
                                        activation='relu', kernel_initializer='he_normal')
            self.bn3 = layers.BatchNormalization()
            self.dropout3 = layers.Dropout(self.DROPOUT)

            self.separable_conv4 = layers.SeparableConv2D(filters=self.n_filters, kernel_size=(5, 5), strides=self.reduction_stride, padding='same',
                                        activation='relu', kernel_initializer='he_normal')
            self.bn4 = layers.BatchNormalization()
            self.dropout4 = layers.Dropout(self.DROPOUT)

            self.maxpooling5 = layers.MaxPooling2D(pool_size=(3, 3), strides=self.reduction_stride, padding='same')

            self.averagepooling6 = layers.AveragePooling2D(pool_size=(3, 3), strides=self.reduction_stride, padding='same')

            self.archlayer1 = ArchLayer(num_inputs=8)
        elif self.reduction_stride == 2:

            self.conv1 = layers.Conv2D(filters=self.n_filters, kernel_size=(3, 3), strides=self.reduction_stride, padding='same',
                                       activation='relu',
                                       kernel_initializer='he_normal')
            self.bn1 = layers.BatchNormalization()
            self.dropout1 = layers.Dropout(self.DROPOUT)

            self.conv2 = layers.Conv2D(filters=self.n_filters, kernel_size=(5, 5), strides=self.reduction_stride, padding='same',
                                       activation='relu',
                                       kernel_initializer='he_normal')
            self.bn2 = layers.BatchNormalization()
            self.dropout2 = layers.Dropout(self.DROPOUT)

            self.separable_conv3 = layers.SeparableConv2D(filters=self.n_filters, kernel_size=(3, 3),
                                                          strides=self.reduction_stride, padding='same',
                                                          activation='relu', kernel_initializer='he_normal')
            self.bn3 = layers.BatchNormalization()
            self.dropout3 = layers.Dropout(self.DROPOUT)

            self.separable_conv4 = layers.SeparableConv2D(filters=self.n_filters, kernel_size=(5, 5),
                                                          strides=self.reduction_stride, padding='same',
                                                          activation='relu', kernel_initializer='he_normal')
            self.bn4 = layers.BatchNormalization()
            self.dropout4 = layers.Dropout(self.DROPOUT)

            self.archlayer1 = ArchLayer(num_inputs=4)

        else:

            self.conv1 = layers.Conv2D(filters=self.n_filters, kernel_size=(3, 3), strides=1, padding='same',
                                       activation='relu',
                                       kernel_initializer='he_normal')
            self.bn1 = layers.BatchNormalization()
            self.dropout1 = layers.Dropout(self.DROPOUT)

            self.conv2 = layers.Conv2D(filters=self.n_filters, kernel_size=(5, 5), strides=1, padding='same',
                                       activation='relu',
                                       kernel_initializer='he_normal')
            self.bn2 = layers.BatchNormalization()
            self.dropout2 = layers.Dropout(self.DROPOUT)

            self.separable_conv3 = layers.SeparableConv2D(filters=self.n_filters, kernel_size=(3, 3), strides=1,
                                                          padding='same',
                                                          activation='relu', kernel_initializer='he_normal')
            self.bn3 = layers.BatchNormalization()
            self.dropout3 = layers.Dropout(self.DROPOUT)

            self.separable_conv4 = layers.SeparableConv2D(filters=self.n_filters, kernel_size=(5, 5), strides=1,
                                                          padding='same',
                                                          activation='relu', kernel_initializer='he_normal')
            self.bn4 = layers.BatchNormalization()
            self.dropout4 = layers.Dropout(self.DROPOUT)

            self.archlayer1 = ArchLayer(num_inputs=4)



    def call(self, inputs, training=True):

        if self.reduction_stride == 1:

            y1 = self.conv1(inputs)
            y1 = self.bn1(y1)
            y1 = self.dropout1(y1)

            y2 = self.conv2(inputs)
            y2 = self.bn2(y2)
            y2 = self.dropout2(y2)

            y3 = self.separable_conv3(inputs)
            y3 = self.bn3(y3)
            y3 = self.dropout3(y3)

            y4 = self.separable_conv4(inputs)
            y4 = self.bn2(y4)
            y4 = self.dropout4(y4)

            y5 = self.maxpooling5(inputs)

            y6 = self.averagepooling6(inputs)

            y7 = inputs

            y8 = tf.zeros(tf.shape(inputs))

            node_outputs = self.archlayer1([y1, y2, y3, y4, y5, y6, y7, y8])


        elif self.reduction_stride == 2:

            y1 = self.conv1(inputs)
            y1 = self.bn1(y1)
            y1 = self.dropout1(y1)

            y2 = self.conv2(inputs)
            y2 = self.bn2(y2)
            y2 = self.dropout2(y2)

            y3 = self.separable_conv3(inputs)
            y3 = self.bn3(y3)
            y3 = self.dropout3(y3)

            y4 = self.separable_conv4(inputs)
            y4 = self.bn2(y4)
            y4 = self.dropout4(y4)

            node_outputs = self.archlayer1([y1, y2, y3, y4])


        else:

            y1 = self.conv1(inputs)
            y1 = self.bn1(y1)
            y1 = self.dropout1(y1)

            y2 = self.conv2(inputs)
            y2 = self.bn2(y2)
            y2 = self.dropout2(y2)

            y3 = self.separable_conv3(inputs)
            y3 = self.bn3(y3)
            y3 = self.dropout3(y3)

            y4 = self.separable_conv4(inputs)
            y4 = self.bn2(y4)
            y4 = self.dropout4(y4)

            node_outputs = self.archlayer1([y1, y2, y3, y4])

        return node_outputs


class ArchLayer(tf.keras.layers.Layer):
    def __init__(self, num_inputs, **kwargs):
        super(ArchLayer, self).__init__(**kwargs)
        self.num_inputs = num_inputs

    def build(self, input_shape):
        self.kernel = self.add_variable("arch_kernel",
                                        shape=[int(self.num_inputs)],
                                        dtype=tf.float32,
                                        initializer=tf.keras.initializers.zeros())   # zeros initialization, equal for each one
        super(ArchLayer, self).build(input_shape)

    def call(self, inputs):
        temp_kernel = tf.nn.softmax(self.kernel)
        outputs = 0
        # print(temp_kernel)
        for i in range(len(inputs)):
            outputs +=  temp_kernel[i] * inputs[i]
        return outputs




if __name__ =='__main__':

    import numpy as np
    from keras.utils import to_categorical
    from keras.datasets import cifar10

    tf.enable_eager_execution()

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    # Import and parse datasets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    input_shape = x_train.shape[1:]

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    subtract_pixel_mean = True
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    # Convert class vectors to binary class matrices.
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Construct keras model
    input_shape = (32, 32, 3)  # CIFAR-10

    n_cells = 8
    alphas = np.random.rand(n_cells, n_cells, 8)

    n_filters = 16
    reduction_channels = False


    inputs = np.random.rand(100, 32, 32, 3)
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

    outputs = dart_cnn_model(num_classes, alphas, n_filters, reduction_channels)

    print(outputs(inputs))










































