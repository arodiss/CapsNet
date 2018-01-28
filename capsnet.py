import tensorflow as tf
import numpy as np
import data
from time import time


caps1_n_maps = 16
caps1_n_caps = caps1_n_maps * 6 * 6  # primary capsules
caps1_n_dims = 6
caps2_n_caps = 10  # digit capsules
caps2_n_dims = 8
routing_rounds = 3


def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector


def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)


def get_performance(labels_per_class):
    tf.reset_default_graph()
    X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")
    y = tf.placeholder(shape=[None, 10], dtype=tf.int64, name="y")
    batch_size = tf.shape(X)[0]

    conv1_params = {
        "filters": caps1_n_maps * caps1_n_dims,
        "kernel_size": 9,
        "strides": 1,
        "padding": "valid",
        "activation": tf.nn.relu,
    }

    conv2_params = {
        "filters": caps1_n_maps * caps1_n_dims,
        "kernel_size": 9,
        "strides": 2,
        "padding": "valid",
        "activation": tf.nn.relu
    }

    conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
    conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
    caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw")
    caps1_output = squash(caps1_raw, name="caps1_output")

    # this initialization works quite well, but there probably is a better one
    W_init = tf.random_normal(
        shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
        stddev=0.1, dtype=tf.float32, name="W_init"
    )
    W = tf.Variable(W_init, name="W")
    W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

    caps1_output_expanded = tf.expand_dims(caps1_output, -1, name="caps1_output_expanded")
    caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_tile")
    caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1], name="caps1_output_tiled")
    caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")

    # routing
    raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1], dtype=np.float32, name="raw_weights")
    routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")
    weighted_predictions = tf.multiply(routing_weights, caps2_predicted, name="weighted_predictions")
    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weighted_sum")
    caps2_output_round_1 = squash(weighted_sum, axis=-2, name="caps2_output_round_1")

    def has_more_rounds(previous_round_output, rounds_passed):
        return tf.less(rounds_passed, routing_rounds)

    def do_routing_round(previous_round_output, rounds_passed):
        previous_round_output_tiled = tf.tile(previous_round_output, [1, caps1_n_caps, 1, 1, 1])
        agreement = tf.matmul(caps2_predicted, previous_round_output_tiled, transpose_a=True)
        raw_weights_current_round = tf.add(raw_weights, agreement)
        routing_weights_current_round = tf.nn.softmax(raw_weights_current_round, dim=2)
        weighted_predictions_current_round = tf.multiply(routing_weights_current_round, caps2_predicted)
        weighted_sum_current_round = tf.reduce_sum(weighted_predictions_current_round, axis=1, keep_dims=True)
        return squash(weighted_sum_current_round, axis=-2), tf.add(rounds_passed, 1)

    rounds_passed = tf.constant(1)
    caps2_output = tf.while_loop(has_more_rounds, do_routing_round, [caps2_output_round_1, rounds_passed], swap_memory=True)[0]

    y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")
    y_proba_softmax = tf.nn.softmax(tf.squeeze(y_proba, axis=[1, 3]), name="y_proba_softmax")

    # Original paper uses argmax + so-called "margin loss" to allow detection of multiple digits,
    #    but here we don't need it and use ordinary softmax + cross-entropy loss instead
    # Original paper also adds reconstruction loss, but here it is just skipped
    loss = tf.reduce_mean(-tf.reduce_sum(tf.cast(y, tf.float32) * tf.log(y_proba_softmax), reduction_indices=[1]))

    correct = tf.equal(
        tf.argmax(y, axis=-1),
        tf.argmax(y_proba_softmax, axis=-1),
        name="correct"
    )
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    # original paper also offers reconstruction loss, but here it is skipped

    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss, name="training_op")

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        epochs_wo_improvement = 0
        epochs_wo_improvement_lr = 0
        current_lr = 0.001
        best_loss = None
        # calling generator directly would give new dataset every epoch - unfair advantage over CNN
        train_data = list(data.get_train_data_generator(labels_per_class, 16))

        start = time()
        while True:
            accumulated_loss = 0
            train_accs = []
            for batch in train_data:
                X_batch, y_batch = batch
                _, loss_train, accuracy_train = sess.run(
                    [training_op, loss, accuracy],
                    feed_dict={
                        X: X_batch,
                        y: y_batch,
                        learning_rate: current_lr
                    }
                )
                accumulated_loss += loss_train
                train_accs.append(accuracy_train)

            if best_loss is None or best_loss > accumulated_loss * 1.001:
                best_loss = float(accumulated_loss)
                epochs_wo_improvement = 0
                epochs_wo_improvement_lr = 0
            else:
                epochs_wo_improvement += 1
                epochs_wo_improvement_lr += 1

            # reduce LR on plateau
            if epochs_wo_improvement_lr >= 5:
                epochs_wo_improvement_lr = 0
                current_lr = current_lr * 0.3

            # early stopping
            if epochs_wo_improvement >= 10:
                break

        train_time = time() - start
        train_acc = np.mean(train_accs)

        start = time()
        accuracies = []
        num_test = 0
        for batch in data.get_test_data_generator(16):
            X_batch, y_batch = batch
            num_test += len(y_batch)
            batch_accuracy = sess.run(
                [accuracy],
                feed_dict={
                    X: X_batch,
                    y: y_batch
                }
            )
            accuracies.append(batch_accuracy)
        test_time = time() - start

        # mean is not exactly correct, but with abundance of val data this does not really matter
        test_acc = np.mean(accuracies)

        return train_acc, test_acc, train_time, 1000 * test_time / num_test
