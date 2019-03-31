""" Utilities
"""
import numpy as np
import random
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical



def loss(model, x, y):
    predictions = model(x)  # In the model, no softmax operation is performed
    loss = tf.losses.softmax_cross_entropy(logits=predictions, onehot_labels=y)  # softmax operations performed here
    return predictions, loss  # return predictions to avoid repeat calculation in calculating accuracy

def grad(model, x, y, vars):
    with tf.GradientTape() as grad_tape:
        _, loss_value = loss(model, x, y)
        grads = grad_tape.gradient(loss_value, vars)
    return grads

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

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
    x_train = x_train_[:40000]
    y_train = y_train_[:40000]
    x_val = x_train_[40000:]
    y_val = y_train_[40000:]

    if debugging_process:
        x_train = x_train[:4000]
        y_train = y_train[:4000]
        x_val = x_val[:1000]
        y_val = y_val[:1000]
        print("_____________________________________________")
        print("Script debugging process")
        print("_____________________________________________")
    else:
        print("_____________________________________________")
        print("Model training process.")
        print("_____________________________________________")

    print('x_train shape:', x_train.shape)
    print('x_val shape:', x_val.shape)

    return (x_train, y_train), (x_val, y_val)


def parse_datasets_ratio(debugging_process, train_ratio):
    """
Args:
    debugging_process: if true, number of training samples is 1000, else 25000.
    train_ratio: ratio of samples as validation dataset.
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
    num_train = int(x_train_.shape[0] * train_ratio)
    x_train = x_train_[:num_train]
    y_train = y_train_[:num_train]
    x_val = x_train_[num_train:]
    y_val = y_train_[num_train:]

    if debugging_process:
        x_train = x_train[:1000]
        y_train = y_train[:1000]
        x_val = x_val[:1000]
        y_val = y_val[:1000]
        print("_____________________________________________")
        print("Script debugging process")
        print("_____________________________________________")
    else:
        print("_____________________________________________")
        print("Model training process.")
        print("_____________________________________________")

    print('x_train shape:', x_train.shape)
    print('x_val shape:', x_val.shape)

    return (x_train, y_train), (x_val, y_val)







def datasets_shuffle_split(x, y, train_ratio):

    combined = list(zip(x, y))
    random.shuffle(combined)
    x[:], y[:] = zip(* combined)
    num_samples = int(x.shape[0] * train_ratio)

    return x[:num_samples], y[:num_samples]


def cutout(input_img):
    p = 0.5
    s_l = 0.02
    s_h = 0.4
    r_1 = 0.3
    r_2 = 1 / 0.3
    v_l = 0
    v_h = 1
    pixel_level = False

    img_h, img_w, img_c = input_img.shape
    p_1 = np.random.rand()

    if p_1 > p:
        return input_img

    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)

        if left + w <= img_w and top + h <= img_h:
            break

    if pixel_level:
        c = np.random.uniform(v_l, v_h, (h, w, img_c))
    else:
        c = np.random.uniform(v_l, v_h)

    input_img[top:top + h, left:left + w, :] = c

    return input_img

