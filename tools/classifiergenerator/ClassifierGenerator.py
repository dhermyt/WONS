import os

from configuration.appsettings import Settings
from definitions import DATASETS_LOCAL_DIR
from tools.SentimentAnalysisToolbox import SentimentAnalysisToolbox
from tools.classifiergenerator.toolsettings import Settings as ToolSettings
from tools.classifiergenerator.toolsettings import SettingsEncoder as ToolSettingsEncoder

if __name__ == '__main__':
    toolSettings = ToolSettings()
    s = Settings()
    s.filterStopwords = toolSettings.FILTER_STOPWORDS
    s.maxNgrams = toolSettings.MAX_NGRAMS
    s.topInformativeFeaturesPercentile = toolSettings.TOP_INFORMATIVE_FEATURES_PERCENTILE
    s.lemmatizerType = toolSettings.LEMMATIZER_TYPE
    lemmatizer = None
    if s.lemmatizerType is not None:
        lemmatizer = s.lemmatizerType()
        lemmatizer.initialize()
    dest_dataset = os.path.join(DATASETS_LOCAL_DIR, toolSettings.WONS_DATASET_SOURCE)
    classifier = toolSettings.CLASSIFIER_TYPE()
    classifier.train(dest_dataset, s)
    classifier.save(toolSettings.WONS_CLASSIFIER_DESTINATION_NAME)
    toolbox = SentimentAnalysisToolbox()
    toolbox.save_json(toolSettings.WONS_CLASSIFIER_DESTINATION_NAME, toolSettings, ToolSettingsEncoder)

