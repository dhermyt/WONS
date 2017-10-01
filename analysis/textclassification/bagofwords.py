import functools

from nltk import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import nltk.corpus
import re
import definitions

INVALID_TOKEN_PATTERN = r'^[!%"%\*\(\)\+,&#-\.\$/\d:;\?\<\>\=@\[\]].*'
NEGATION_TOKEN_PATTERN = r'^nie$'

def get_stopwords_list():
    return list(nltk.corpus.stopwords.words('polish'))

def filter_stopwords(words):
    polish_stopwords = get_stopwords_list()
    return [w for w in words if w not in polish_stopwords]


def filter_custom_set(words, custom_set):
    r = re.compile(custom_set)
    words = list(filter(lambda w: not r.match(w), words))
    return words


def include_significant_bigrams(words, score_fn=BigramAssocMeasures.likelihood_ratio, n=100):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return list(words + bigrams)


def get_all_lowercase(words):
    return [x.lower() for x in words]


def get_bag_of_words(words):
    return dict([(word, True) for word in words])


def mark_negations(words):
    add_negation_suffix = False
    r_negation = re.compile(NEGATION_TOKEN_PATTERN)
    r_stopword = re.compile(INVALID_TOKEN_PATTERN)
    for index, item in enumerate(words):
        if (r_stopword.match(item)):
            add_negation_suffix = False
            continue
        if (r_negation.match(item)):
            add_negation_suffix = True
            continue
        if (add_negation_suffix):
            words[index] = words[index] + "_NEG"
    return words


def get_processed_bag_of_words(text, lemmatizer, settings):
    words = nltk.tokenize.word_tokenize(text, 'polish')
    words = get_all_lowercase(words)
    if lemmatizer is not None:
        words = [lemmatizer.get_lemma(word) for word in words]
    if (settings.FILTER_STOPWORDS):
        words = filter_stopwords(words)
    words = mark_negations(words)
    words = filter_custom_set(words, INVALID_TOKEN_PATTERN)
    if settings.MAX_FEATURES > 0:
        words = words[:settings.MAX_FEATURES]
    words = functools.reduce(lambda x, y: x + y,
                             [words if n == 1 else list([' '.join(ngram) for ngram in ngrams(words, n)]) for n in
                              range(1, settings.MAX_NGRAMS + 1)])
    return get_bag_of_words(words)
