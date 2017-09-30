import codecs
import gzip
import json
import os
import pickle

from analysis import corpus
from analysis.textclassification import bagofwords
from definitions import SENTIMENT_ANALYSIS_DATA_DIR


class SentimentAnalysisToolbox:
    def load_data_set(self, datasetname, settings, feature_detector=bagofwords.get_processed_bag_of_words):
        categorized_set = corpus.load_categorized_data(datasetname)
        feature_set = corpus.label_features_from_corpus(categorized_set, feature_detector)
        selected_feature_set = corpus.select_most_informative_features(feature_set,
                                                                       settings.TOP_INFORMATIVE_FEATURES_PERCENTILE)
        return corpus.split_label_features(selected_feature_set, settings.TRAIN_TEST_SPLIT, True)

    def load_features(self, datasetname, settings):
        lemmatizer = None
        if settings.lemmatizerType is not None:
            lemmatizer = settings.lemmatizerType()
            lemmatizer.initialize()
        return self.load_data_set(datasetname, settings,
                                  lambda x: bagofwords.get_processed_bag_of_words(x, lemmatizer,
                                                                                  settings)
                                  )

    def load_binary(self, filename, compressed=False):
        filepath = os.path.join(SENTIMENT_ANALYSIS_DATA_DIR, filename)
        f = gzip.open(filepath, 'rb') if compressed else open(filepath, 'rb')
        component = pickle.load(f)
        f.close()
        return component

    def save_binary(self, filename, component, compressed=False):
        filepath = os.path.join(SENTIMENT_ANALYSIS_DATA_DIR, filename)
        f = gzip.open(filepath, 'wb') if compressed else open(filepath, 'wb')
        pickle.dump(component, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def save_json(self, filename, component, encoder=None):
        filepath = os.path.join(SENTIMENT_ANALYSIS_DATA_DIR, "{}.json".format(filename))
        f = codecs.open(filepath, 'w', encoding='utf-8')
        f.write(json.dumps(component.__dict__, ensure_ascii=False, cls=encoder))
        f.close()

    def load_json(self, filename, component, decoder=None):
        filepath = os.path.join(SENTIMENT_ANALYSIS_DATA_DIR, "{}.json".format(filename))
        f = codecs.open(filepath, 'r', encoding='utf-8')
        component.__dict__ = json.load(f, cls=decoder)
        return component
