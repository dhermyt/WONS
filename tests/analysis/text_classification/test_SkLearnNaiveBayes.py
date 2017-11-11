# -*- coding: utf-8 -*-

import unittest

from analysis.textclassification.SklearnClassifierFactory import SklearnClassifierFactory
from configuration.appsettings import Settings
from local_definitions import DATASET_TEST_COMMENTS


class Test_SkLearnNaiveBayes(unittest.TestCase):
    def test_get_labels(self):
        classifier = SklearnClassifierFactory.SklearnMultinomialNB()
        s = Settings()
        s.maxNgrams = 1
        s.filterStopwords = True
        s.maxFeatures = 3
        classifier.train(DATASET_TEST_COMMENTS, s)
        self.assertEqual(len(classifier.get_labels()), 3)

    def test_get_name(self):
        classifier = SklearnClassifierFactory.SklearnMultinomialNB()
        self.assertEqual(classifier.get_name(), 'MultinomialNB')


if __name__ == '__main__':
    unittest.main()
