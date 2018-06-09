import os
import glob as gl
import numpy as np

class Data(object):
    def __init__(self, train, test, valid):
        self.train = train
        self.test = test
        self.valid = valid

def process_data(data_list, data_length, data_stride, batch_size):
    feature_list = []
    label_list = []
    name_list = []

    for data in data_list:
        loaded_data = np.load(data)
        num_data = (loaded_data[0].shape[0]-data_length)/data_stride + 1
        for i in range(num_data):
            feature = loaded_data[0][i*data_stride:i*data_stride+data_length]
            feature = feature.reshape(batch_size, feature.shape[0]/batch_size, feature.shape[1])
            label = loaded_data[1]
            name = data.split('/')[-2] + '/' + data.split('/')[-1]

            feature_list.append(feature)
            label_list.append(label)
            name_list.append(name)

    feature_list = np.asarray(feature_list)
    feature_list = np.transpose(feature_list, (0, 1, 3, 2))
    label_list = np.asarray(label_list)
    name_list = np.asarray(name_list)

    return feature_list, label_list, name_list

def process_data_phoneme(data_list, data_length, data_stride, batch_size):
    feature_list = []
    phoneme_list = []
    label_list = []
    name_list = []

    for data in data_list:
        loaded_data = np.load(data)
        num_data = (loaded_data[0].shape[0]-data_length)/data_stride + 1
        for i in range(num_data):
            feature = loaded_data[0][i*data_stride:i*data_stride+data_length]
            feature = feature.reshape(batch_size, feature.shape[0]/batch_size, feature.shape[1])
            phoneme = loaded_data[1][i*data_stride:i*data_stride+data_length]
            phoneme = phoneme.reshape(batch_size, phoneme.shape[0]/batch_size, phoneme.shape[1])
            label = loaded_data[-1]
            name = data.split('/')[-2] + '/' + data.split('/')[-1]

            feature_list.append(feature)
            phoneme_list.append(phoneme)
            label_list.append(label)
            name_list.append(name)

    feature_list = np.asarray(feature_list)
    phoneme_list = np.asarray(phoneme_list)

    feature_new = np.concatenate((feature_list, phoneme_list), axis=3)
    feature_new = np.transpose(feature_new, (0, 1, 3, 2))
    label_list = np.asarray(label_list)
    name_list = np.asarray(name_list)

    return feature_new, label_list, name_list

def normalize_data(data, means, stds):
    normalized_data = np.zeros(data.shape)
    for i in range(data.shape[-1]):
        normalized_data[:,:,:,i] = (data[:,:,:,i] - means[i])/stds[i]

    return normalized_data

def preprocess(dataset_dir, train_ratio, data_length, data_stride, batch_size, normalize=False):
    train_list = gl.glob(os.path.join(os.path.join(dataset_dir, 'train'), '*.npy'))
    train_size = int(np.round(len(train_list))*train_ratio)

    train_list = gl.glob(os.path.join(os.path.join(dataset_dir, 'train'), '*.npy'))[:train_size]
    test_list = gl.glob(os.path.join(os.path.join(dataset_dir, 'test'), '*.npy'))
    valid_list = gl.glob(os.path.join(os.path.join(dataset_dir, 'train'), '*.npy'))[train_size+1:]

    x_train, y_train, name_train = process_data(train_list, data_length, data_stride, batch_size)
    x_test, y_test, name_test = process_data(test_list, data_length, data_stride, batch_size)
    x_valid, y_valid, name_valid = process_data(valid_list, data_length, data_stride, batch_size)

    if normalize == True:
        x_train_reshape = x_train.reshape(x_train.shape[0]*x_train.shape[1]*x_train.shape[2], x_train.shape[-1])
        means = np.mean(x_train_reshape, axis=0)
        stds = np.std(x_train_reshape, axis=0)

        x_train = normalize_data(x_train, means, stds)
        x_test = normalize_data(x_test, means, stds)
        x_valid = normalize_data(x_valid, means, stds)

    return Data(x_train, x_test, x_valid), Data(y_train, y_test, y_valid), Data(name_train, name_test, name_valid)

def preprocess_phoneme(dataset_dir, train_ratio, data_length, data_stride, batch_size, normalize=False):
    train_list = gl.glob(os.path.join(os.path.join(dataset_dir, 'train'), '*.npy'))
    train_size = int(np.round(len(train_list))*train_ratio)

    train_list = gl.glob(os.path.join(os.path.join(dataset_dir, 'train'), '*.npy'))[:train_size]
    test_list = gl.glob(os.path.join(os.path.join(dataset_dir, 'test'), '*.npy'))
    valid_list = gl.glob(os.path.join(os.path.join(dataset_dir, 'train'), '*.npy'))[train_size+1:]

    x_train, y_train, name_train = process_data_phoneme(train_list, data_length, data_stride, batch_size)
    x_test, y_test, name_test = process_data_phoneme(test_list, data_length, data_stride, batch_size)
    x_valid, y_valid, name_valid = process_data_phoneme(valid_list, data_length, data_stride, batch_size)

    if normalize == True:
        x_train_reshape = x_train.reshape(x_train.shape[0]*x_train.shape[1]*x_train.shape[2], x_train.shape[-1])
        means = np.mean(x_train_reshape, axis=0)
        stds = np.std(x_train_reshape, axis=0)

        x_train = normalize_data(x_train, means, stds)
        x_test = normalize_data(x_test, means, stds)
        x_valid = normalize_data(x_valid, means, stds)

    return Data(x_train, x_test, x_valid), Data(y_train, y_test, y_valid), Data(name_train, name_test, name_valid)

def evaluate(prediction, annotation):
    tn = 0.0
    fp = 0.0
    fn = 0.0
    tp = 0.0

    for i in range(annotation.shape[0]):
        if (annotation[i] == 0 and prediction[i] == 0):
            tn += 1
        elif (annotation[i] == 0 and prediction[i] != 0):
            fp += 1
        elif (annotation[i] != 0 and prediction[i] == 0):
            fn += 1
        elif (annotation[i] != 0 and prediction[i] != 0):
            tp += 1

    accuracy = (tp + tn)/(tn + fp + fn + tp)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    fscore = 2*precision*recall/(precision + recall)

    return accuracy, precision, recall, fscore
