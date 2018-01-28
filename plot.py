import matplotlib.pyplot as plt
import numpy as np
from cnn import get_performance as get_cnn_performance
from capsnet import get_performance as get_capsnet_performance


def get_cnn_score(train_size):
    return average([
        get_cnn_performance(train_size),
        get_cnn_performance(train_size),
        get_cnn_performance(train_size)
    ])


def get_capsnet_score(train_size):
    return average([
        get_capsnet_performance(train_size),
        get_capsnet_performance(train_size),
        get_capsnet_performance(train_size)
    ])


def average(ps):
    # facepalm
    q = []
    w = []
    e = []
    r = []
    for p in ps:
        nq, nw, ne, nr = p
        q.append(nq)
        w.append(nw)
        e.append(ne)
        r.append(nr)
    return np.mean(q), np.mean(w), np.mean(e), np.mean(r)


def plot_performace(train_size_powers):
    cnn_times_test = []
    cnn_times_train = []
    caps_times_test = []
    caps_times_train = []
    cnn_accs_train = []
    cnn_accs_test = []
    caps_accs_train = []
    caps_accs_test = []
    print(
        '{} settings to evaluate. Each next is going to take exponentially longer than previous'
        .format(len(train_size_powers))
    )
    for index, train_size_power in enumerate(train_size_powers):
        train_size = 10 ** train_size_power
        new_cnn_acc_train, new_cnn_acc_test, new_cnn_time_train, new_cnn_time_test = get_cnn_score(train_size)
        print('   ... evaluated case #{} for CNN...'.format(index + 1))
        new_capsnet_acc_train, new_capsnet_acc_test, new_capsnet_time_train, new_capsnet_time_test = get_capsnet_score(train_size)
        print('   ... evaluated case #{} for CapsNet...'.format(index + 1))
        cnn_accs_train.append(new_cnn_acc_train)
        cnn_accs_test.append(new_cnn_acc_test)
        cnn_times_train.append(new_cnn_time_train)
        cnn_times_test.append(new_cnn_time_test)

        caps_accs_train.append(new_capsnet_acc_train)
        caps_accs_test.append(new_capsnet_acc_test)
        caps_times_train.append(new_capsnet_time_train)
        caps_times_test.append(new_capsnet_time_test)
        print('Completed case #{} out of {}'.format(index + 1, len(train_size_powers)))

    print('Raw data:')
    print(caps_accs_train, caps_accs_test, cnn_accs_train, cnn_accs_test)
    print('')
    print(caps_times_test, cnn_times_test)
    print('')
    print(caps_times_train, cnn_times_train)
    print('-----------------')

    plt.subplot(1, 3, 1)
    plt.title('Accuracy')
    plt.plot(train_size_powers, caps_accs_train, 'b', label='CapsNet train')
    plt.plot(train_size_powers, caps_accs_test, 'c', label='CapsNet test')
    plt.plot(train_size_powers, cnn_accs_train, 'r', label='CNN train')
    plt.plot(train_size_powers, cnn_accs_test, 'm', label='CNN test')
    plt.legend()
    plt.xlabel('Training dataset size\n(10 ** x cases per class)')

    plt.subplot(1, 3, 2)
    plt.title('Inference time, ms per case')
    plt.plot(train_size_powers, caps_times_test, 'b', label='CapsNet')
    plt.plot(train_size_powers, cnn_times_test, 'r', label='CNN')
    plt.legend()
    plt.xlabel('Training dataset size\n(10 ** x cases per class)')

    plt.subplot(1, 3, 3)
    plt.title('Training time, seconds')
    plt.plot(train_size_powers, caps_times_train, 'b', label='CapsNet')
    plt.plot(train_size_powers, cnn_times_train, 'r', label='CNN')
    plt.legend()
    plt.xlabel('Training dataset size\n(10 ** x cases per class)')

    plt.show()
    plt.close()


plot_performace([v / 2 for v in list(range(0, 7))])
