class Settings:
    def __init__(self):
        self.FILTER_STOPWORDS = True
        self.LEMMATIZER_TYPE = None
        self.MAX_NGRAMS = 1
        self.MAX_FEATURES = 150
        self.TOP_INFORMATIVE_FEATURES_PERCENTILE = 1.0
        self.TRAIN_TEST_SPLIT = 0.75
        self.CLASSIFIER_TYPE = None