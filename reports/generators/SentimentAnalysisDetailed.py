import codecs
import json
import os
from itertools import groupby

import definitions
import local_definitions
from analysis import corpus
from analysis.textclassification import bagofwords
from tools.SentimentAnalysisToolbox import SentimentAnalysisToolbox


class SentimentAnalysisDetailedReportGenerator:
    def generate(self, dataset, classifier, settings):
        lemmatizer = None
        if settings.lemmatizerType is not None:
            lemmatizer = settings.lemmatizerType()
            lemmatizer.initialize()
        train_set, test_set = classifier.train(dataset, settings)
        reports = []
        for key, group in groupby(test_set, lambda x: x[1]):
            items = list(group)
            for item in items:
                report = {}
                report['text'] = item[0]
                report['sentiment'] = key
                featureset = classifier.get_vectorizer.build_analyzer()(item[0])
                report['feature_set'] = dict(
                    (feature, classifier.classify(feature, 'neutral')) for feature in featureset)
                report['classification'] = classifier.classify(item[0], 'neutral')
                report['probability'] = {}
                report['isClassifiedCorrectly'] = report['sentiment'] == report['classification']
                # dist = classifier.prob_classify(featureset)
                # for label in dist.samples():
                #     report['probability'][label] = round(dist.prob(label), 5)

                if report['sentiment'] == report['classification']:
                    continue
                reports.append(report)

        filename = '{}.json'.format(type(self).__name__)
        f = codecs.open(os.path.join(definitions.REPORTS_LOCAL_DIR, filename), 'w', encoding='utf-8')
        f.write(json.dumps(reports, ensure_ascii=False))
        f.close()
