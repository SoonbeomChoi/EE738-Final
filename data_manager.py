import os
import glob as gl
import numpy as np

class Data(object):
    def __init__(self, train, test, valid):
        self.train = train
        self.test = test
        self.valid = valid

def process_data(data_list, data_length, batch_size):
    feature_list = []
    label_list = []

    x_dim = np.load(data_list[0])[0].shape[1]

    for data in data_list:
        loaded_data = np.load(data)
        feature = np.zeros((data_length, x_dim))
        if loaded_data[0].shape[0] > data_length:
            feature[:data_length] = loaded_data[0][:data_length]
        else:
            feature[:loaded_data[0].shape[0]] = loaded_data[0]

        feature = feature.reshape(batch_size, feature.shape[0]/batch_size, feature.shape[1])
        label = loaded_data[1]

        feature_list.append(feature)
        label_list.append(label)

    feature_list = np.asarray(feature_list)
    feature_list = np.transpose(feature_list, (0, 1, 3, 2))
    label_list = np.asarray(label_list)

    return feature_list, label_list

def preprocess(dataset_dir, train_ratio, data_length, batch_size):
    train_list = gl.glob(os.path.join(os.path.join(dataset_dir, 'train'), '*.npy'))
    train_size = int(np.round(len(train_list))*train_ratio)

    train_list = gl.glob(os.path.join(os.path.join(dataset_dir, 'train'), '*.npy'))[:train_size]
    test_list = gl.glob(os.path.join(os.path.join(dataset_dir, 'test'), '*.npy'))
    valid_list = gl.glob(os.path.join(os.path.join(dataset_dir, 'train'), '*.npy'))[train_size+1:]

    x_train, y_train = process_data(train_list, data_length, batch_size)
    x_test, y_test = process_data(test_list, data_length, batch_size)
    x_valid, y_valid = process_data(valid_list, data_length, batch_size)

    return Data(x_train, x_test, x_valid), Data(y_train, y_test, y_valid)

def evaluate(prediction, annotation):
    accuracy = 1-np.count_nonzero(prediction - annotation)/float(prediction.shape[0])

    return accuracy
