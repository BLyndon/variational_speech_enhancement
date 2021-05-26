import os
import numpy as np

from scipy.io import wavfile
from scipy.signal import stft

import pickle

import tensorflow as tf


def save_data(obj, file_path):
    '''
    Save object as pickle-file.
    '''
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print('Save pickle file to "{}"'.format(file_path))


def load_data(file_path):
    '''
    Load pickle-file.
    '''
    print('Load pickle file "{}"'.format(file_path))
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def get_wav_paths(wav_dir):
    '''
    - Collect wav-file paths and return them as a list.
    '''
    file_paths = list()
    print('Get wave-file paths from\n "{}"'.format(wav_dir))
    for root, _, files in os.walk(wav_dir):
        for file in files:
            if file.endswith('.wav'):
                file_paths.append(os.path.join(root, file))
    return list(set(file_paths))


def write_transform_from_filelist(file_paths, fname, nperseg=1024):
    """
    - Read in wav-files from a file list.
    - Transform to STFT-domain.
    - Save a dictionary containing all information about the wav-files.
    """
    data = {'set': [], 'dialect': [],
            'speaker': [], 'file': [],
            'X': [], 'f': [], 't': []}

    N = len(file_paths)
    for n, file_path in enumerate(file_paths):
        print("{}/{}".format(n+1, N), end='\r')

        fs, x = wavfile.read(file_path)
        f, t, X = stft(x, fs, nperseg=nperseg)
        assert fs == int(16e3), "Expected: fs = 16e3."

        path, file_name = os.path.split(file_path)
        path, speaker = os.path.split(path)
        path, dialect = os.path.split(path)
        path, set_type = os.path.split(path)

        data['set'].append(set_type)
        data['dialect'].append(dialect)
        data['speaker'].append(speaker)
        data['file'].append(file_name)
        data['X'].append(X)
        data['f'].append(f)
        data['t'].append(t)

    save_data(data, fname)


def load_write_dataset_split(data_dir, nperseg=1024):
    '''
    - Load and preprocess training and testing data.
    - Store all data information in a dictionary.
    - Save the dictionary as a pickle-file.
    '''
    for folder in ['train', 'test']:
        fname = os.path.join(data_dir, '{}.pckl'.format(folder))
        if os.path.isfile(fname):
            print('File "{}.pckl" already exists, continue!'.format(folder))
            continue
        else:
            paths = get_wav_paths(os.path.join(data_dir, folder.upper()))

            print('{} shape: {}'.format(folder, len(paths)))

            write_transform_from_filelist(
                paths, fname, nperseg=nperseg)


def get_dataset_from_file(file_path, batch_size=128, buffer_size=-1):
    '''
    - Load data from pickle file.
    - Create TF-Dataset of STFT in cartesian coordinates.
    '''
    data_info = load_data(file_path)

    X = np.hstack(data_info['X']).T

    print("Dataset shape: {}".format(X.shape))
    ds_X = tf.data.Dataset.from_tensor_slices(X)

    if buffer_size > 0:
        ds_X = ds_X.shuffle(buffer_size)
    return ds_X.batch(batch_size)


if __name__ == "__main__":
    pass
