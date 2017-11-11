import pickle
import re

import nltk
from nltk.classify.util import accuracy

from analysis.textclassification import bagofwords
from definitions import DATA_DIR
from helpers.file import File
from tools.SentimentAnalysisToolbox import SentimentAnalysisToolbox

nltk.data.path.append(DATA_DIR)


class NltkClassifierWrapper(object):
    __classifier = None
    __classifierType = None

    def __init__(self, classifierType):
        self.__classifierType = classifierType

    @property
    def is_initialized(self):
        return self.__classifier is not None

    def load(self, filename=None):
        toolbox = SentimentAnalysisToolbox()
        if filename is None:
            filename = '{}.classifier'.format(self.get_name())
        self.__classifier = toolbox.load_binary(filename)

    def save(self, filename=None):
        toolbox = SentimentAnalysisToolbox()
        if filename is None:
            filename = '{}.classifier'.format(self.get_name())
        toolbox.save_binary(filename, self.__classifier)

    def train(self, datasetname, settings):
        lemmatizer = None
        if settings.lemmatizerType is not None:
            lemmatizer = settings.lemmatizerType()
            lemmatizer.initialize()
        toolbox = SentimentAnalysisToolbox()
        train_set, test_set = toolbox.load_data_set(datasetname, settings,
                                                    lambda x: bagofwords.get_processed_bag_of_words(x, lemmatizer,
                                                                                                    settings))
        self.__classifier = self.__classifierType.train(train_set)
        return train_set, test_set

    def classify(self, featureset, default_label=None):
        if not self.is_initialized:
            return
            # if default_label is not None:
            #     try:
            #         prob_classify = self.prob_classify(featureset)
            #         prob_samples = [round(prob_classify.prob(sample), 2) for sample in prob_classify.samples()]
            #         if (prob_samples.count(prob_samples[0]) == len(prob_samples)):
            #             return default_label
            #     except AttributeError:
            #         pass
            # if 'PYCHARM_HOSTED' in os.environ:
            #     print(featureset)
            #     for sample in prob_classify.samples():
            #         print(sample, prob_classify.prob(sample))
            # if prob_classify is not None and default_label is not None:
            #     prob_samples = [round(prob_classify.prob(sample), 2) for sample in prob_classify.samples()]
            #     if (prob_samples.count(prob_samples[0]) == len(prob_samples)):
            #         return default_label
        # File.append('nltk.txt', str(featureset) + '\n')
        return self.__classifier.classify(featureset)

    def prob_classify(self, featureset):
        if not self.is_initialized:
            return None
        if "prob_classify" in dir(self.__classifierType):
            return self.__classifier.prob_classify(featureset)
        return None

    def get_accuracy(self, testset):
        if not self.is_initialized:
            return 0
        return accuracy(self.__classifier, testset)

    def most_informative_features(self, count):
        if not self.is_initialized:
            return
        if "most_informative_features" in dir(self.__classifierType):
            return self.__classifier.most_informative_features(count)
        return None

    def get_labels(self):
        if not self.is_initialized:
            return
        return self.__classifier.labels()

    def get_name(self):
        pattern = re.compile(r'\s+')
        return re.sub(pattern, '', str(self.__classifierType))
