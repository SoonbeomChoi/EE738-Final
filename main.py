import os
import numpy as np

import data_manager
import model_archive
from model_wrapper import Wrapper

DATASET_DIR = './Dataset/mfcc_phoneme/'
EXPORT_DIR = './Export/phoneme_no_norm_conv1d_256_128_64_32/'

DEVICE = 1 # 0 : cpu, 1 : gpu0, 2 : gpu1, ...
NUM_CLASS = 4
DATA_LENGTH = 256
DATA_STRIDE = 32
TRAIN_RATIO = 0.75
NORMALIZE = False

EPOCH = 300
BATCH_SIZE = 1
LEARN_RATE = 0.0001

def main():
    x, y, name = data_manager.preprocess_phoneme(DATASET_DIR, TRAIN_RATIO, DATA_LENGTH, DATA_STRIDE, BATCH_SIZE, NORMALIZE)
    print('Data Loaded')

    acc = data_manager.Data(np.zeros(EPOCH), None, np.zeros(EPOCH))
    loss = data_manager.Data(np.zeros(EPOCH), None, np.zeros(EPOCH))

    model = model_archive.CONV1D2_256_128_64_32(x.train.shape[2], NUM_CLASS)
    wrapper = Wrapper(model, LEARN_RATE)

    print('\n--------- Training Start ---------')

    for epoch in range(EPOCH):
        _, acc.train[epoch], loss.train[epoch] = wrapper.run_model(x.train, y.train, DEVICE, 'train')
        _, acc.valid[epoch], loss.valid[epoch] = wrapper.run_model(x.valid, y.valid, DEVICE, 'eval')

        if wrapper.early_stop(loss.valid[epoch]): break

        print('Epoch [' + str(epoch+1).zfill(int(np.log10(EPOCH))+1) + '/' + str(EPOCH) + ']'
         + ' acc : ' + str(round(acc.train[epoch],4)) + ' - val_acc : ' + str(round(acc.valid[epoch],4))
         + ' | loss : ' + str(round(loss.train[epoch],4)) + ' - val_loss : ' + str(round(loss.valid[epoch],4)))

    print('-------- Training Finished -------')
    pred_test, acc_test, _ = wrapper.run_model(x.test, y.test, DEVICE, 'eval')
    accuracy, precision, recall, fscore = data_manager.evaluate(pred_test, y.test)
    print('\nClassification Accuracy : ' + str(round(acc_test,4)))
    print('Binary Accuracy : ' + str(round(accuracy,4)))
    print('Precision : ' + str(round(precision,4)))
    print('Recall : ' + str(round(recall,4)))
    print('F SCORE : ' + str(round(fscore,4)))
    wrapper.export(EXPORT_DIR, x.test, y.test, pred_test)
    print('\nFiles exported to ' + os.path.abspath(EXPORT_DIR))

if __name__ == '__main__':
    main()
