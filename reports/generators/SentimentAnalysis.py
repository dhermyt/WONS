from collections import defaultdict
from itertools import groupby


class SentimentAnalysisReportGenerator:
    def generate(self, dataset, classifier, settings):
        train_set, test_set = classifier.train(dataset, settings)
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
                class_result = classifier.classify(item[0], 'unknown')
                if class_result == key:
                    correct += 1
                else:
                    mistakes[class_result + '_' + key] += 1
            report['accuracy'][key] = correct / len(items)
        report['mostInformativeFeatures'] = classifier.most_informative_features(20)
        report['mostMistakes'] = sorted(mistakes.items(), key=lambda x: x[1], reverse=True)[:3]
        return report
