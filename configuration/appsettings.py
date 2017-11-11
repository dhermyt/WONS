class Settings:
    def __init__(self):
        self.filterStopwords = True
        self.lemmatizerType = None
        self.maxNgrams = 1
        self.maxFeatures = 150
        self.topInformativeFeaturesPercentile = 1.0
        self.trainTestSplit = 0.75