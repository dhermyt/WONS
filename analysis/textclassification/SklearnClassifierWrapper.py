import re

from sklearn.feature_extraction.text import TfidfVectorizer

from analysis import corpus
from analysis.textclassification import bagofwords
from helpers.file import File
from tools.SentimentAnalysisToolbox import SentimentAnalysisToolbox


class SklearnClassifierWrapper(object):
    __classifier = None
    __classifierType = None
    __vectorizer = None

    def __init__(self, classifierType):
        self.__classifierType = classifierType

    @property
    def get_vectorizer(self):
        return self.__vectorizer

    @property
    def is_initialized(self):
        return self.__classifier is not None

    def load(self):
        toolbox = SentimentAnalysisToolbox()
        filename = '{}.classifier'.format(self.get_name())
        self.__classifier = toolbox.load_binary(filename)
        filename = '{}.vectorizer'.format(self.get_name())
        self.__vectorizer = toolbox.load_binary(filename)

    def save(self):
        toolbox = SentimentAnalysisToolbox()
        filename = '{}.classifier'.format(self.get_name())
        toolbox.save_binary(filename,self.__classifier)
        filename = '{}.vectorizer'.format(self.get_name())
        toolbox.save_binary(filename, self.__vectorizer)

    def train(self, datasetname, settings):
        categorized_set = corpus.load_categorized_data(datasetname)
        train_set, test_set = corpus.split_label_features(categorized_set, settings.trainTestSplit, True)
        self.__vectorizer = self.create_vectorizer(settings)
        train_vectors = self.__vectorizer.fit_transform([data for data, classinfo in train_set])
        train_classes = [classinfo for data, classinfo in train_set]
        self.__classifier = self.__classifierType()
        self.__classifier = self.__classifier.fit(train_vectors, train_classes)
        return train_set, test_set

    def create_vectorizer(self, settings):
        polish_stopwords = bagofwords.get_stopwords_list() if settings.filterStopwords else None
        return TfidfVectorizer(stop_words=polish_stopwords,
                               ngram_range=(1, settings.maxNgrams),
                               tokenizer=self.create_tokenizer)

    def create_tokenizer(self, doc):
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        return bagofwords.mark_negations(token_pattern.findall(doc))

    def classify(self, featureset, default_label=None):
        if not self.is_initialized:
            return
        if default_label is not None:
            try:
                prob_classify = self.__classifier.predict_proba(self.__vectorizer.transform([featureset]))[0]
                prob_samples = [round(sample, 2) for sample in prob_classify]
                if (prob_samples.count(prob_samples[0]) == len(prob_samples)):
                    return default_label
            except AttributeError:
                pass
        # test = self.__vectorizer.transform([featureset])
        # File.append('sklearn.txt', str(self.__vectorizer.transform([featureset])) + '\n')
        return self.__classifier.predict(self.__vectorizer.transform([featureset]))[0]

    def prob_classify(self, featureset):
        if not self.is_initialized:
            return None
        if "prob_classify" in dir(self.__classifierType):
            return self.__classifier.prob_classify(featureset)
        return None

    def get_accuracy(self, testset):
        if not self.is_initialized:
            return 0

        test_vectors = self.__vectorizer.transform([data for data, classinfo in testset])
        test_classes = [classinfo for data, classinfo in testset]
        return self.__classifier.score(test_vectors, test_classes)

    def most_informative_features(self, count):
        if not self.is_initialized:
            return
        if "most_informative_features" in dir(self.__classifierType):
            return self.__classifier.most_informative_features(count)
        return None

    def get_labels(self):
        if not self.is_initialized:
            return
        return self.__classifier.classes_

    def get_name(self):
        pattern = re.compile(r'\s+')
        return re.sub(pattern, '', str(self.__classifierType.__name__))
