import os
import codecs
import random

import nltk.data
from itertools import groupby

from nltk import FreqDist, ConditionalFreqDist, BigramAssocMeasures

import definitions
from definitions import DATA_DIR

nltk.data.path.append(DATA_DIR)


def load_categorized_data(name):
    categorized_data = []
    categories = [item for item in os.listdir(name) if os.path.isdir(os.path.join(name, item))]
    for category in categories:
        path = os.path.join(name, category)
        dataFiles = [file for file in os.listdir(path) if file.endswith('.txt')]
        for dataFile in dataFiles:
            f = codecs.open(os.path.join(path, dataFile), 'r', encoding='utf-8')
            categorized_data.extend([(line.strip(), category) for line in f.readlines()])
            f.close()
    return categorized_data


def label_features_from_corpus(corpus, feature_detector):
    label_features = []
    for data, category in corpus:
        features = feature_detector(data)
        if (len(features) > 0):
            label_features.append((feature_detector(data), category))
    return label_features


def split_label_features(label_features, split=0.75, shuffle=False):
    train_set = []
    test_set = []
    for key, group in groupby(label_features, lambda x: x[1]):
        feats = list(group)
        # if shuffle:
            # random.shuffle(feats)
        cutoff = int(len(feats) * split)
        train_set.extend([feat for feat in feats[:cutoff]])
        test_set.extend([feat for feat in feats[cutoff:]])
    return train_set, test_set


def select_most_informative_features(feature_set, top_informative_features_percentile):
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()
    for feature_list, groupKey in feature_set:
        for feature in feature_list:
            word_fd[feature] += 1
            label_word_fd[groupKey][feature] += 1
    label_feature_count = dict((label, label_word_fd[label].N()) for label in label_word_fd)
    total_feature_count = sum([label_word_fd[label].N() for label in label_word_fd])
    feature_scores = {}
    for feature in word_fd:
        label_scores = {(label, BigramAssocMeasures.chi_sq(label_word_fd[label][feature],
                                                           (word_fd[feature], label_feature_count[label]),
                                                           total_feature_count)) for label in label_word_fd}
        feature_scores[feature] = sum([score for label, score in label_scores])
    top_features_count = int(round(top_informative_features_percentile * total_feature_count))
    best = sorted(feature_scores, key=lambda x: feature_scores[x], reverse=True)[:top_features_count]
    bestwords = set([w for w in best])
    filtered_feature_set = []
    for feature_list, groupKey in feature_set:
        filtered_feature_set.append((dict([(x, True) for x in feature_list if x in bestwords]), groupKey))
    return filtered_feature_set
