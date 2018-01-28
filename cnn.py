from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.backend import clear_session
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from data import get_train_data, get_test_data
from time import time


image_size = 28
num_channels = 1
num_labels = 10

batch_size = 16
patch_size = 3
num_hidden_1 = 400
num_hidden_2 = 300
learning_rate = 0.001
dropout_rate_1 = 0.1
dropout_rate_2 = 0.2
dropout_rate_3 = 0.2
num_filters_1 = 16
num_filters_2 = 32
num_filters_3 = 64


def get_performance(labels_per_class):
    clear_session()
    model = Sequential()
    model.add(
        Conv2D(
            num_filters_1,
            (patch_size, patch_size),
            activation='relu',
            padding="valid",
            input_shape=(image_size, image_size, num_channels),
            kernel_initializer='he_uniform'
        )
    )
    model.add(Conv2D(num_filters_2, (patch_size, patch_size), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(pool_size=(patch_size, patch_size)))
    model.add(Conv2D(num_filters_3, (patch_size, patch_size), activation='relu', kernel_initializer='he_uniform'))
    model.add(Flatten())
    model.add(Dropout(dropout_rate_1))
    model.add(Dense(num_hidden_1, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate_2))
    model.add(Dense(num_hidden_2, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate_3))
    model.add(Dense(num_labels, kernel_initializer='he_uniform', activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=learning_rate),
        metrics=['accuracy']
    )

    x_train, y_train = get_train_data(labels_per_class)
    x_test, y_test = get_test_data()

    start = time()
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=1000,  # will be aborted by early stopping when necessary
        callbacks=[
            # should use validation data instead, but, well, I just don't
            EarlyStopping(monitor='loss', patience=10, verbose=False, mode='min'),
            ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5, verbose=False, mode='min'),
        ],
        verbose=False
    )
    train_acc = history.history['acc'][-1]
    train_time = time() - start

    start = time()
    test_acc = model.evaluate(x_test, y_test, verbose=False)[1]
    test_time = time() - start

    return train_acc, test_acc, train_time, 1000 * test_time / len(y_test)
