# -*- coding: utf-8 -*-

import unittest
import nltk.data
import os
import codecs
from nltk.metrics import BigramAssocMeasures
from definitions import DATA_DIR

nltk.data.path.append(DATA_DIR)


class Test_nltkSamples(unittest.TestCase):
    def test_sent_tokenize(self):
        text = 'Litwo! Ojczyzno moja! Ty jesteś jak zdrowie. Ile cię trzeba cenić, ten tylko chodzić zwykła z któremi przy boku rzuciwszy wzrok stryja ku studni.'
        tokenizer = nltk.data.load('tokenizers\punkt\polish.pickle')
        tokens = tokenizer.tokenize(text)
        self.assertEqual(len(tokens), 4)

    def test_word_tokenize(self):
        text = 'Ile cię trzeba cenić, ten tylko chodzić zwykła z któremi przy boku rzuciwszy wzrok stryja ku studni.'
        tokens = nltk.tokenize.word_tokenize(text)
        self.assertEqual(len(tokens), 19)

    def test_word_tokenize_without_punctuation(self):
        text = 'Ile cię trzeba cenić, ten tylko chodzić zwykła z któremi przy boku rzuciwszy wzrok stryja ku studni.'
        tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(text)
        self.assertEqual(len(tokens), 17)

    def test_filter_stopwords(self):
        text = 'Ile cię trzeba cenić, ten tylko chodzić zwykła z któremi przy boku rzuciwszy wzrok stryja ku studni.'
        words = nltk.tokenize.word_tokenize(text)
        polish_stopwords = set(nltk.corpus.stopwords.words('polish'))
        filtered_words = [word for word in words if word not in polish_stopwords]
        self.assertEqual(len(filtered_words), 12)

    def test_wordnet_synsets(self):
        synsets = nltk.corpus.wordnet.synsets('Politechnika')
        self.assertEqual(len(synsets), 2)

    def test_synsets_wup(self):
        w1 = nltk.corpus.wordnet.synsets('kot')[0]
        w2 = nltk.corpus.wordnet.synsets('drapak')[0]
        self.assertTrue(isinstance(w2.wup_similarity(w1), float))

    def test_collocations(self):
        input_file = codecs.open(os.path.join(os.path.dirname(__file__), 'sample_data', 'pantadeusz_ksiega1.txt'), 'r',
                                 encoding='utf-8')
        raw_text = input_file.read()
        words = nltk.tokenize.word_tokenize(raw_text)
        polish_stopwords = set(nltk.corpus.stopwords.words('polish'))
        filter_stops = lambda w: len(w) < 3 or w in polish_stopwords
        bcf = nltk.collocations.BigramCollocationFinder.from_words(words)
        bcf.apply_word_filter(filter_stops)
        self.assertEqual(len(bcf.nbest(BigramAssocMeasures.likelihood_ratio, 3)), 3)


if __name__ == '__main__':
    unittest.main()
