import codecs
import os

from analysis.lemmatizers.dblookup import DbLookupLemmatizer
from analysis.textclassification import bagofwords
from analysis.textclassification.NltkClassifierFactory import NltkClassifierFactory
from configuration.appsettings import Settings
from local_definitions import DATASET_UNKNOWN_DATA, DATASET_PUDELEK_COMMENTS
from tools.SentimentAnalysisToolbox import SentimentAnalysisToolbox


def cls():
    os.system('cls')


class SentimentAnalysisVotingMachine:
    def read_one_line(self):
        filename = 'query.txt'
        filepath = os.path.join(DATASET_UNKNOWN_DATA, filename)
        f = codecs.open(filepath, 'r', 'utf-8')
        lines = f.readlines()
        f.close()
        if len(lines) == 0:
            return None
        f = codecs.open(filepath, 'w', 'utf-8')
        f.writelines(lines[1:])
        f.close()
        return lines[0]

    def read_one_unknown_line(self, classifier):
        filename = 'query.txt'
        filepath = os.path.join(DATASET_UNKNOWN_DATA, filename)
        f = codecs.open(filepath, 'r', 'utf-8')
        lines = f.readlines()
        f.close()
        if len(lines) == 0:
            return None
        for index, line in enumerate(lines):
            featureset = bagofwords.get_processed_bag_of_words(line, lemmatizer, s)
            prob_classify = classifier.prob_classify(featureset)
            prob_samples = [round(prob_classify.prob(sample), 2) for sample in prob_classify.samples()]
            if (max(prob_samples) < 0.4):
                f = codecs.open(filepath, 'w', 'utf-8')
                f.writelines([line_in for index_in, line_in in enumerate(lines) if index_in != index])
                f.close()
                return line


        return None

    def read_one_unknown_line2(self, classifier):
        filename = 'query.txt'
        filepath = os.path.join(DATASET_UNKNOWN_DATA, filename)
        f = codecs.open(filepath, 'r', 'utf-8')
        lines = f.readlines()
        f.close()
        if len(lines) == 0:
            return None
        for index, line in enumerate(lines):
            featureset = bagofwords.get_processed_bag_of_words(line, lemmatizer, s)
            unknown_features = 0
            for feature in featureset:
                prob_classify = classifier.prob_classify(dict([(feature, True)]))
                prob_samples = [round(prob_classify.prob(sample), 2) for sample in prob_classify.samples()]
                if (prob_samples.count(prob_samples[0]) == len(prob_samples)):
                    unknown_features += 1
            if (unknown_features / len(featureset) > 0.8):
                f = codecs.open(filepath, 'w', 'utf-8')
                f.writelines([line_in for index_in, line_in in enumerate(lines) if index_in != index])
                f.close()
                return line

        return None

    def read_one_known_line(self, classifier, class_key):
        filename = 'query.txt'
        filepath = os.path.join(DATASET_UNKNOWN_DATA, filename)
        f = codecs.open(filepath, 'r', 'utf-8')
        lines = f.readlines()
        f.close()
        if len(lines) == 0:
            return None
        for index, line in enumerate(lines):
            featureset = bagofwords.get_processed_bag_of_words(line, lemmatizer, s)
            key = classifier.classify(featureset, 'neutral')
            if (key == class_key):
                f = codecs.open(filepath, 'w', 'utf-8')
                f.writelines([line_in for index_in, line_in in enumerate(lines) if index_in != index])
                f.close()
                return line

        return None

    def create_stats(self, classifier):
        filename = 'query.txt'
        stat_filename = 'query_stat.txt'
        filepath = os.path.join(DATASET_UNKNOWN_DATA, filename)
        f = codecs.open(filepath, 'r', 'utf-8')
        lines = f.readlines()
        f.close()
        if len(lines) == 0:
            return None
        stats = []
        for index, line in enumerate(lines):
            featureset = bagofwords.get_processed_bag_of_words(line, lemmatizer, s)
            key = classifier.classify(featureset, 'neutral')
            prob_classify = classifier.prob_classify(featureset)
            prob_stats = {}
            for label in prob_classify.samples():
                prob_stats[label] = round(prob_classify.prob(label), 5)
            stats.append(key + " " + str(prob_stats) + " " + line)

        filepath = os.path.join(DATASET_UNKNOWN_DATA, stat_filename)
        f = codecs.open(filepath, 'w', 'utf-8')
        f.writelines(stats)
        f.close()
        return None

    def add_line_to_dataset(self, line, class_key, dataset):
        filepath = os.path.join(dataset, class_key, 'data.txt')
        f = codecs.open(filepath, 'a', 'utf-8')
        f.write(line)
        f.close()


if __name__ == '__main__':
    vm = SentimentAnalysisVotingMachine()
    toolbox = SentimentAnalysisToolbox()
    s = Settings()
    s.filterStopwords = True
    s.maxNgrams = 3
    s.topInformativeFeaturesPercentile = 1.0
    s.lemmatizerType = DbLookupLemmatizer
    lemmatizer = None
    if s.lemmatizerType is not None:
        lemmatizer = s.lemmatizerType()
        lemmatizer.initialize()
    dest_dataset = DATASET_PUDELEK_COMMENTS
    classifier = NltkClassifierFactory.SklearnMultinomialNB()
    classifier.train(dest_dataset, s)
    # vm.create_stats(classifier)
    while True:
        cls()
        line = vm.read_one_line()
        if (line == None):
            break
        featureset = bagofwords.get_processed_bag_of_words(line, lemmatizer, s)
        class_key = classifier.classify(featureset, 'unknown')
        # if(class_key != 'unknown'):
            # vm.add_line_to_dataset(line, class_key, dest_dataset)
            # continue
        print("Text (" + class_key + "):\n")

        print(line)
        vote = input("Określ sentyment:")
        if (vote == '0'):
            continue
        class_vote = 'neutral'
        if (vote == '1'):
            class_vote = 'negative'
        elif (vote == '3'):
            class_vote = 'positive'

        vm.add_line_to_dataset(line, class_vote, dest_dataset)
