import os
import sys

import numpy as np
import librosa

DATASET_DIR = './SRS_DB/'
EXTRACTED_DIR = './Dataset/phoneme_mfcc/'

def labeling(file_path):
    label = 0
    if 'Word01' in file_path:
        label = 1
    elif 'Word05' in file_path:
        label = 2
    elif 'Word09' in file_path:
        label = 3

    return label

def log_spectrogram(file_path, fft_size=1024, hop_size=160):
    y, sr = librosa.load(file_path)
    S = np.abs(librosa.stft(y, n_fft=fft_size, hop_length=hop_size))
    feature = librosa.power_to_db(S**2).T

    return feature


def mfcc(file_path, fft_size=1024, hop_size=160, n_mels=128, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=fft_size, hop_length=hop_size, n_mels=n_mels)
    feature = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc).T

    return feature

def feature_extraction(file_path, dataset_dir=DATASET_DIR, extracted_dir=EXTRACTED_DIR):
    feature = mfcc(file_path)
    label = labeling(file_path)

    audio_filename = os.path.basename(file_path)
    numpy_filename = audio_filename.replace('.wav', '.npy')

    feature_n_label = np.array((feature.astype(np.float32), label), dtype=object)

    if 'Train' in file_path:
        dataset_subdir = 'train'
    elif 'Test' in file_path:
        dataset_subdir = 'test'
    np.save(os.path.join(extracted_dir, dataset_subdir, numpy_filename), feature_n_label)

def feature_extraction_phoneme(file_path, dataset_dir=DATASET_DIR, extracted_dir=EXTRACTED_DIR):
    feature = mfcc(file_path)
    label = labeling(file_path)

    phoneme_path = file_path.replace('SRS_DB', 'Phoneme') + '.scores'
    phoneme = np.loadtxt(phoneme_path, skiprows=1)

    phoneme_new =np.zeros((feature.shape[0], phoneme.shape[1]))
    if phoneme.shape[0] < feature.shape[0]:
        phoneme_new[:phoneme.shape[0],:] = phoneme
        phoneme_new[phoneme.shape[0]:,:] = phoneme[-1,:]
    else:
        phoneme_new = phoneme[:feature.shape[0],:]

    audio_filename = os.path.basename(file_path)
    numpy_filename = audio_filename.replace('.wav', '.npy')

    feature = np.array((feature.astype(np.float32), phoneme_new.astype(np.float32), label), dtype=object)

    if 'Train' in file_path:
        dataset_subdir = 'train'
    elif 'Test' in file_path:
        dataset_subdir = 'test'

    np.save(os.path.join(extracted_dir, dataset_subdir, numpy_filename), feature)

def main():
    if not os.path.exists(os.path.join(EXTRACTED_DIR, 'train')):
        os.makedirs(os.path.join(EXTRACTED_DIR, 'train'))

    if not os.path.exists(os.path.join(EXTRACTED_DIR, 'test')):
        os.makedirs(os.path.join(EXTRACTED_DIR, 'test'))

    num_wav = 0
    for path, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith('.wav'):
                num_wav += 1

    file_cnt = 0
    pertenmile = 0
    for path, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith('.wav'):
                file_cnt += 1
                file_path = os.path.join(path, file)
                feature_extraction_phoneme(file_path)
                if pertenmile < 10000*file_cnt/num_wav:
                    pertenmile = 10000*file_cnt/num_wav
                    print('[' + str(pertenmile/100.0) + '%] is done')
                    sys.stdout.write("\033[F")

    print('finished')

if __name__ == '__main__':
    main()
