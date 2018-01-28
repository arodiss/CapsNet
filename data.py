import numpy as np
import keras
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


def get_train_data(labels_per_class):
    x_test_chunk = []
    y_test_chunk = []
    current_label_nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in np.random.permutation(len(y_train)):
        current_label = np.argmax(y_train[i])
        if current_label_nums[current_label] < labels_per_class:
            current_label_nums[current_label] += 1
            x_test_chunk.append(x_train[i])
            y_test_chunk.append(y_train[i])
        if sum(current_label_nums) == labels_per_class * 10:
            break
    return np.array(x_test_chunk), np.array(y_test_chunk)


def get_test_data():
    return x_test, y_test


# as a side effect these two functions undo one-hot encoding
def get_train_data_generator(labels_per_class, batch_size):
    current_index = 0
    x, y = get_train_data(labels_per_class)
    while current_index < len(x):
        yield x[current_index:current_index+batch_size], y[current_index:current_index+batch_size]
        current_index += batch_size


def get_test_data_generator(batch_size):
    current_index = 0
    x, y = get_test_data()
    while current_index < len(x):
        yield x[current_index:current_index+batch_size], y[current_index:current_index+batch_size]
        current_index += batch_size
