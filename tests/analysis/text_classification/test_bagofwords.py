# -*- coding: utf-8 -*-

import unittest

from nltk import NaiveBayesClassifier

from analysis.lemmatizers.dblookup import DbLookupLemmatizer
from analysis.textclassification import bagofwords
from analysis.textclassification.NltkClassifierWrapper import NltkClassifierWrapper
from configuration.appsettings import Settings
from local_definitions import DATASET_TEST_COMMENTS
from tools.SentimentAnalysisToolbox import SentimentAnalysisToolbox


class Test_BagOfWords(unittest.TestCase):
    def test_negative_sentence(self):
        text = "Nie lubię go. Jest niedojrzały."
        s = Settings()
        lemmatizer = DbLookupLemmatizer()
        lemmatizer.initialize()
        tokens = bagofwords.get_processed_bag_of_words(text, lemmatizer, s)
        pass


if __name__ == '__main__':
    unittest.main()
