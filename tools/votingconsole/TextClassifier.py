import os

from analysis.lemmatizers.dblookup import DbLookupLemmatizer
from analysis.textclassification import bagofwords
from analysis.textclassification.NltkClassifierFactory import NltkClassifierFactory
from definitions import DATASETS_LOCAL_DIR
from tools.votingconsole.toolsettings import ToolSettings


class TextClassifier:
    __classifier = None
    __lemmatizer = None
    __settings = None

    def initialize(self):
        toolSettings = ToolSettings()
        toolSettings.FILTER_STOPWORDS = True
        toolSettings.MAX_NGRAMS= 3
        toolSettings.TOP_INFORMATIVE_FEATURES_PERCENTILE= 1.0
        toolSettings.LEMMATIZER_TYPE = DbLookupLemmatizer
        self.__lemmatizer = None
        if toolSettings.LEMMATIZER_TYPE is not None:
            self.__lemmatizer = toolSettings.LEMMATIZER_TYPE()
            self.__lemmatizer.initialize()
        dest_dataset = os.path.join(DATASETS_LOCAL_DIR, toolSettings.WONS_CLASSIFIER_SOURCE)
        self.__classifier = NltkClassifierFactory.SklearnMultinomialNB()
        self.__classifier.train(dest_dataset, toolSettings)
        self.__settings = toolSettings

    def matches(self, line):
        if self.__settings.CLASSIFIER_MATCH_METHOD == "all":
            return True
        if self.__settings.CLASSIFIER_MATCH_METHOD == "only_unknown":
            return self.is_unknown_line(line)
        if self.__settings.CLASSIFIER_MATCH_METHOD == "only_uncertain":
            return self.is_uncertain_line(line)

    def is_unknown_line(self, line):
        linesToPrint = []
        linesToPrint.append("line: {}".format(line))
        featureset = bagofwords.get_processed_bag_of_words(line, self.__lemmatizer, self.__settings)
        linesToPrint.append("featureset: {}".format(featureset))
        if len(featureset) == 0:
            [print(x) for x in linesToPrint]
            return True
        prob_classify = self.__classifier.prob_classify(featureset)
        for sample in prob_classify.samples():
            linesToPrint.append(sample)
            linesToPrint.append(prob_classify.prob(sample))
        unknown_features = 0
        for feature in featureset:
            prob_classify = self.__classifier.prob_classify(dict([(feature, True)]))
            prob_samples = [round(prob_classify.prob(sample), 2) for sample in prob_classify.samples()]
            if (prob_samples.count(prob_samples[0]) == len(prob_samples)):
                unknown_features += 1
        isUnknown = (unknown_features / len(featureset)) > 0.8
        if isUnknown:
            [print(x) for x in linesToPrint]
        return isUnknown

    def is_uncertain_line(self, line):
        linesToPrint = []
        linesToPrint.append("line: {}".format(line))
        featureset = bagofwords.get_processed_bag_of_words(line, self.__lemmatizer, self.__settings)
        linesToPrint.append("featureset: {}".format(featureset))
        if len(featureset) == 0:
            [print(x) for x in linesToPrint]
            return True
        prob_classify = self.__classifier.prob_classify(featureset)
        for sample in prob_classify.samples():
            linesToPrint.append("{}: {}".format(sample, prob_classify.prob(sample)))
        prob_classify = self.__classifier.prob_classify(featureset)
        prob_samples = [round(prob_classify.prob(sample), 2) for sample in prob_classify.samples()]
        isUncertain = max(prob_samples) < self.__settings.CLASSIFIER_MATCH_UNCERTAINTY_THRESHOLD
        if isUncertain:
            linesToPrint.append("my type: {}".format(prob_classify.max()))
            [print(x) for x in linesToPrint]
        return isUncertain