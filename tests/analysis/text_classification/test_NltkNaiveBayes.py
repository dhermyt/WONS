# -*- coding: utf-8 -*-

import unittest

from nltk import NaiveBayesClassifier

from analysis.textclassification.NltkClassifierWrapper import NltkClassifierWrapper
from configuration.appsettings import Settings
from local_definitions import DATASET_TEST_COMMENTS
from tools.SentimentAnalysisToolbox import SentimentAnalysisToolbox


class Test_NaiveBayesSentimentAnalysis(unittest.TestCase):
    def test_get_labels(self):
        classifier = NltkClassifierWrapper(NaiveBayesClassifier)
        s = Settings()
        s.maxNgrams = 1
        s.filterStopwords = True
        s.lemmatizerType = None
        s.maxFeatures = 3
        s.topInformativeFeaturesPercentile = 0.03
        classifier.train(DATASET_TEST_COMMENTS, s)
        self.assertEqual(len(classifier.get_labels()), 3)

    def test_get_name(self):
        classifier = NltkClassifierWrapper(NaiveBayesClassifier)
        self.assertEqual(classifier.get_name(), '<class\'nltk.classify.naivebayes.NaiveBayesClassifier\'>')


if __name__ == '__main__':
    unittest.main()
