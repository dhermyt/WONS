from collections import defaultdict
from itertools import groupby
import local_definitions
from tools.SentimentAnalysisToolbox import SentimentAnalysisToolbox


class SentimentAnalysisCrossCorpora:
    def generate(self, dataset, classifier, settings):
        toolbox = SentimentAnalysisToolbox()
        settings.trainTestSplit = 1.0
        toolbox.train(classifier, dataset, settings)
        test_set, temp = toolbox.load_features(local_definitions.DATASET_PUDELEK_COMMENTS, settings)
        report = {}
        mistakes = defaultdict(int)
        report['classifier'] = classifier.get_name()
        report['dataset'] = dataset
        report['settings'] = settings.__dict__
        report['totalAccuracy'] = classifier.get_accuracy(test_set)
        report['accuracy'] = {}
        for key, group in groupby(test_set, lambda x: x[1]):
            correct = 0
            items = list(group)
            for item in items:
                class_result = classifier.classify(item[0])
                if class_result == key:
                    correct += 1
                else:
                    mistakes[class_result + '_' + key] += 1
            report['accuracy'][key] = correct / len(items)
        report['mostInformativeFeatures'] = classifier.most_informative_features(20)
        report['mostMistakes'] = sorted(mistakes.items(), key=lambda x: x[1], reverse=True)[:3]
        return report
