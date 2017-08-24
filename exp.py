import numpy as np
import os
import sys
from six.moves import cPickle as pickle
from hashlib import sha1

PICKLE_FILE = './data/notMNIST.pickle'
OUTPUT_FILE = './data/notMNIST_refined.pickle'

def pretty_print(data, with_index=True):
    for key, value in enumerate(data):
        if with_index:
            print('{} - {}'.format(key, value))
        else:
            print('{}'.format(value))

def get_original(dataset):
    train_dataset = dataset['train_dataset']
    train_labels = dataset['train_labels']
    valid_dataset = dataset['valid_dataset']
    valid_labels = dataset['valid_labels']
    test_dataset = dataset['test_dataset']
    test_labels = dataset['test_labels']
    return train_dataset, train_labels, valid_dataset, valid_labels, \
            test_dataset, test_labels

def hash_data(D):
    return {int(idx): sha1(data).hexdigest() for idx, data in enumerate(D)}

def get_intersection(dictA, dictB):
    """Returns numpy array in sorted order that contains
    overlapped elements between two dicts
    """
    A = np.array([d for d in dictA.values()])
    B = np.array([d for d in dictB.values()])
    return np.intersect1d(A, B)

def remove_overlap(original, labels, overlaps, hash):
    print('original shape {}'.format(original.shape))
    indices = [] # indices of elements to be remove from original
    for overlap in overlaps:
        for search in overlap:
            index = list(hash.keys())[list(hash.values()).index(search)]
            # index = get_k_by_v(hash, search)
            indices.append(index)
    data = np.array([data for index, data in enumerate(original) if index not in indices])
    labels = np.array([label for index, label in enumerate(labels) if index not in indices])
    return data, labels

def save(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
    print('Pickling {}'.format(OUTPUT_FILE))
    try:
        f = open(OUTPUT_FILE, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print('Pickling completed!')
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    
def main():
    with open(PICKLE_FILE, 'rb') as f:
        dataset = pickle.load(f)

        # Load train, val and test sets
        train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_original(dataset)
        print(train_labels)

        # Hash them
        train_hash = hash_data(train_dataset)
        valid_hash = hash_data(valid_dataset)
        test_hash = hash_data(test_dataset)

        train_valid = get_intersection(train_hash, valid_hash)
        train_test = get_intersection(train_hash, test_hash)
        test_valid = get_intersection(test_hash, valid_hash)

        print('Eliminating overlapped items from train set')
        train_set, train_set_labels = remove_overlap(
            train_dataset, train_labels, [train_valid, train_test], train_hash)

        print('Eliminating overlapped items from valid set')
        valid_set, valid_set_labels = remove_overlap(
            valid_dataset, valid_labels, [train_valid, test_valid], valid_hash)

        print('Eliminating overlapped items from test set')
        test_set, test_set_labels = remove_overlap(
            test_dataset, test_labels, [train_test, test_valid], test_hash)

        save(
            train_set, train_set_labels,
            valid_set, valid_set_labels,
            test_set, test_set_labels
        )

if __name__ == '__main__':
    main()
