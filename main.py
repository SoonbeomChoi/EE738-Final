import numpy as np

import data_manager
import model_archive
from model_wrapper import Wrapper

DATASET_DIR = './Dataset/mfcc/'
EXPORT_DIR = './Export/'

DEVICE = 1 # 0 : cpu, 1 : gpu0, 2 : gpu1, ...
DATA_LENGTH = 256
TRAIN_RATIO = 0.75
NUM_CLASS = 4 # 0 : Non-keyword, 1 - 3: Keywords

EPOCH = 200
BATCH_SIZE = 1
LEARN_RATE = 0.0001

def main():
    x, y = data_manager.preprocess(DATASET_DIR, TRAIN_RATIO, DATA_LENGTH, BATCH_SIZE)

    acc = data_manager.Data(np.zeros(EPOCH), None, np.zeros(EPOCH))
    loss = data_manager.Data(np.zeros(EPOCH), None, np.zeros(EPOCH))

    model = model_archive.CNN(x.train.shape[2], NUM_CLASS)
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
    print('\nTest Accuracy : ' + str(round(100*acc_test,2)) + '%')

if __name__ == '__main__':
    main()
