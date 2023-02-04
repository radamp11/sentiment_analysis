'''
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
'''

# Import Module
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from string import punctuation
from collections import Counter


def obtain_data(path, seq_length=200, split_frac=0.8, batch_size=50):
    reviews = []
    labels = []

    read_files(path, reviews, labels)
    vocab_to_int = {w: i + 1 for i, (w, c) in enumerate(count_words(reviews))}

    reviews_int = convert_reviews_to_int(reviews, vocab_to_int)
    # show_data_info(reviews_int)

    encoded_labels = [1 if label == 'positive' else 0 for label in labels]
    encoded_labels = np.array(encoded_labels)

    # getting rid of extremely short reviews
    reviews_len = [len(x) for x in reviews_int]
    reviews_int = [reviews_int[i] for i, l in enumerate(reviews_len) if l > 0]
    encoded_labels = [encoded_labels[i] for i, l in enumerate(reviews_len) if l > 0]

    # and shortening too long reviews
    features = pad_features(reviews_int, seq_length)

    # Preparing data sets: 80% train, 10% valid, 10% test
    len_feat = len(features)

    train_x = features[0: int(split_frac * len_feat)]
    train_y = encoded_labels[0: int(split_frac * len_feat)]

    remaining_x = features[int(split_frac * len_feat):]
    remaining_y = encoded_labels[int(split_frac * len_feat):]

    valid_x = remaining_x[0: int(len(remaining_x) * 0.5)]
    valid_y = remaining_y[0: int(len(remaining_y) * 0.5)]

    test_x = remaining_x[int(len(remaining_x) * 0.5):]
    test_y = remaining_y[int(len(remaining_y) * 0.5):]

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(np.asarray(train_x)), torch.from_numpy(np.asarray(train_y)))
    valid_data = TensorDataset(torch.from_numpy(np.asarray(valid_x)), torch.from_numpy(np.asarray(valid_y)))
    test_data = TensorDataset(torch.from_numpy(np.asarray(test_x)), torch.from_numpy(np.asarray(test_y)))

    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, vocab_to_int


'''rest of used functions'''


def read_text_file(file_path):
    with open(file_path, 'r', encoding="utf8") as f:
        return f.read()


def define_label(label_int):
    if label_int <= 4:
        return 'negative'
    else:
        return 'positive'


def read_files(path, reviews, labels):
    for file in os.listdir(path):
        if file.endswith(".txt"):
            tmp_label = re.search('_(.*).txt', file)
            label = define_label( int(tmp_label.group(1)) )
            tmp_text_file = read_text_file(path + file).lower()
            review = ''.join([c for c in tmp_text_file if c not in punctuation])
            reviews.append(review)
            labels.append(label)


def count_words(reviews):
    all_text2 = ' '.join(reviews)
    # create a list of words
    words = all_text2.split()
    # Count all the words using Counter Method
    counted_words = Counter(words)

    total_words = len(words)

    return counted_words.most_common(total_words)


def convert_reviews_to_int(reviews, vocab_to_int):
    reviews_int = []
    for rev in reviews:
        r = [vocab_to_int[w] for w in rev.split()]
        reviews_int.append(r)
    return reviews_int


def show_data_info(encoded_reviews):
    reviews_len = [len(x) for x in encoded_reviews]
    pd.Series(reviews_len).hist()
    plt.show()
    print(pd.Series(reviews_len).describe())


def pad_features(reviews_int, seq_length):
    '''
        Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features_tmp = np.zeros((len(reviews_int), seq_length), dtype=int)

    for i, review in enumerate(reviews_int):
        review_len = len(review)

        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len))
            new = zeroes + review
        elif review_len > seq_length:
            new = review[0:seq_length]

        features_tmp[i, :] = np.array(new)

    return features_tmp
