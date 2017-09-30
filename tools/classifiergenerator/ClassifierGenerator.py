import os

from configuration.Encoder import SettingsEncoder
from definitions import DATASETS_LOCAL_DIR
from tools.SentimentAnalysisToolbox import SentimentAnalysisToolbox
from tools.classifiergenerator.toolsettings import ToolSettings

if __name__ == '__main__':
    toolSettings = ToolSettings()
    lemmatizer = None
    if toolSettings.LEMMATIZER_TYPE is not None:
        lemmatizer = toolSettings.LEMMATIZER_TYPE()
        lemmatizer.initialize()
    dest_dataset = os.path.join(DATASETS_LOCAL_DIR, toolSettings.WONS_DATASET_SOURCE)
    classifier = toolSettings.CLASSIFIER_TYPE()
    classifier.train(dest_dataset, toolSettings)
    classifier.save(toolSettings.WONS_CLASSIFIER_DESTINATION_NAME)
    toolbox = SentimentAnalysisToolbox()
    toolbox.save_json(toolSettings.WONS_CLASSIFIER_DESTINATION_NAME, toolSettings, SettingsEncoder)

