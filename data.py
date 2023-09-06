import numpy as np
from torchvision import datasets


def preprocess(x):
    x = np.float64(x)
    x = (x / 255 - 0.5) * 2
    x = np.clip(x, -1, 1)
    return x


def load_data(args):
    # Dataset: mnist, cifar10, kdd99, custom
    # Ano_class: 1 actual class label of the dataset
    if args.dataset == 'mnist':
        return get_mnist(args.ano_class)
    elif args.dataset == 'cifar10':
        return get_cifar10(args.ano_class)


def get_mnist(ano_class):
    '''
    There are 2 classes: normal and anomalous
    - Training data: x_train: 80% of data/images from normal classes
    - Validation data: x_val: 5% of data/images from normal classes + 25% of data/images from anomalous classes
    - Testing data: x_test: 15% of data/images from normal classes + 75% of data/images from anomalous classes
    '''

    x_tr = datasets.MNIST('.', download=True, train=True)
    y_tr = x_tr.targets.numpy()
    x_tr = x_tr.data.numpy()
    x_tst = datasets.MNIST('.', download=True, train=False)
    y_tst = x_tst.targets.numpy()
    x_tst = x_tst.data.numpy()

    x_total = np.concatenate([x_tr, x_tst])
    y_total = np.concatenate([y_tr, y_tst])
    print(x_total.shape, y_total.shape)

    x_total = x_total.reshape(-1, 28, 28, 1)
    x_total = preprocess(x_total)

    delete = []
    for count, i in enumerate(y_total):
        if i == ano_class:
            delete.append(count)

    ano_data = x_total[delete, :]
    normal_data = np.delete(x_total, delete, axis=0)

    normal_num = normal_data.shape[0]  # Number of data/images of normal classes
    ano_num = ano_data.shape[0]  # Number of data/images of anomalous classes

    del x_total

    x_train = normal_data[:int(0.8*normal_num), ...]

    x_test = np.concatenate(
        (normal_data[int(0.8*normal_num):int(0.95*normal_num), ...], ano_data[:ano_num*3//4]))
    y_test = np.concatenate(
        (np.ones(int(0.95*normal_num)-int(0.8*normal_num)), np.zeros(ano_num*3//4)))

    x_val = np.concatenate(
        (normal_data[int(0.95*normal_num):, ...], ano_data[ano_num*3//4:]))
    y_val = np.concatenate(
        (np.ones(normal_num-int(0.95*normal_num)), np.zeros(ano_num-ano_num*3//4)))

    # swap axes from tf configuration to torch
    x_train = np.swapaxes(x_train, 1, 3)
    x_train = np.swapaxes(x_train, 2, 3)
    x_val = np.swapaxes(x_val, 1, 3)
    x_val = np.swapaxes(x_val, 2, 3)
    x_test = np.swapaxes(x_test, 1, 3)
    x_test = np.swapaxes(x_test, 2, 3)

    return x_train, x_test, y_test, x_val, y_val


def get_cifar10(ano_class):

    X_train = datasets.MNIST('.', download=True, train=True)
    y_train = X_train.targets.numpy()
    X_train = X_train.data.numpy()
    X_test = datasets.MNIST('.', download=True, train=False)
    y_test = X_test.targets.numpy()
    X_test = X_test.data.numpy()

    X_train_anomaly = X_train[np.where(y_train == ano_class)[0]]
    X_train_normal = X_train[np.where(y_train != ano_class)[0]]
    X_test_anomaly = X_test[np.where(y_test == ano_class)[0]]
    X_test_normal = X_test[np.where(y_test != ano_class)[0]]

    X_normal = np.concatenate([X_train_normal, X_test_normal])
    X_anomaly = np.concatenate([X_train_anomaly, X_test_anomaly])

    '''
    Pick 20% of X_normal to be in Test set. Place the remaining in Train + Validation set
    '''
    idx = np.random.choice(X_normal.shape[0], int(0.2 * X_normal.shape[0]), replace=False)
    X_test_normal = np.array([X_normal[i] for i in idx])
    X_trainval_normal = np.array([X_normal[i]
                                 for i in range(X_normal.shape[0]) if i not in idx])

    '''
    Pick 20% of X_anomaly to be in Validation set. Place the remaining in Test Set
    '''
    idx = np.random.choice(X_anomaly.shape[0], int(
        0.2 * X_anomaly.shape[0]), replace=False)
    X_val_anomaly = np.array([X_anomaly[i] for i in idx])
    X_test_anomaly = np.array([X_anomaly[i]
                              for i in range(X_anomaly.shape[0]) if i not in idx])

    '''
    Compute the ratio of normal data and anomaly data in the Test Set
    '''
    ratio = int(X_val_anomaly.shape[0] * X_test_normal.shape[0] / X_test_anomaly.shape[0]
                )  # Compute Validation Ratio between Normal and Anomaly Samples
    idx = np.random.choice(X_trainval_normal.shape[0], ratio, replace=False)
    X_val_normal = np.array([X_trainval_normal[i] for i in idx])

    '''
    Form the Train, Validation and Test Sets
    '''
    X_train = np.array([X_trainval_normal[i]
                       for i in range(X_trainval_normal.shape[0]) if i not in idx])
    X_val = np.concatenate([X_val_normal, X_val_anomaly])
    X_test = np.concatenate([X_test_normal, X_test_anomaly])

    X_train = preprocess(X_train)
    X_val = preprocess(X_val)
    X_test = preprocess(X_test)

    '''
    Prepare Labels
    '''
    y_val_anomaly = [0] * X_val_anomaly.shape[0]  # anomalies are labeled '0'
    y_val_normal = [1] * X_val_normal.shape[0]  # Normal samples are labeled '1'
    y_val = np.asarray(np.concatenate([y_val_normal, y_val_anomaly]))

    y_test_anomaly = [0] * X_test_anomaly.shape[0]
    y_test_normal = [1] * X_test_normal.shape[0]
    y_test = np.asarray(np.concatenate([y_test_normal, y_test_anomaly]))

    # swap axes from tf configuration to torch
    X_train = np.swapaxes(X_train, 1, 3)
    X_train = np.swapaxes(X_train, 2, 3)
    X_val = np.swapaxes(X_val, 1, 3)
    X_val = np.swapaxes(X_val, 2, 3)
    X_test = np.swapaxes(X_test, 1, 3)
    X_test = np.swapaxes(X_test, 2, 3)

    return X_train, X_test, y_test, X_val, y_val

# def get_kdd99():

# def get_custom():
