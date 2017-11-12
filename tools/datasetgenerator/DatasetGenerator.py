import codecs

from tools.DbClient import DbClient
from tools.datasetgenerator.ToolSettings import ToolSettings
from definitions import DATASETS_LOCAL_DIR
import os

key_to_sentiment_mapping = {
    '-1' : 'neg',
    '0' : 'neu',
    '1' : 'pos'
}

if __name__ == '__main__':
    client = DbClient()
    toolSettings = ToolSettings()
    client.connect(toolSettings.DB_USER, toolSettings.DB_PASSWORD, toolSettings.DB_HOST, toolSettings.DB_PORT)
    dataset_names = toolSettings.WONS_DATASET_SOURCE.split(',')
    dataset = []
    for dataset_name in dataset_names:
        dataset += [data for data in client.get_dataset(dataset_name)]
    userVotes = []
    usernames = toolSettings.WONS_USERNAME.split(",")
    for username in usernames:
        user = client.get_user_id(username)
        userVotes += [vote for vote in client.get_user_votes(user)]
    dataset_path = os.path.join(DATASETS_LOCAL_DIR, toolSettings.WONS_DATASET_DEST_NAME)
    voteData = []
    for vote in userVotes:
        for data in dataset:
            if vote["TextDataId"] == data["_id"]:
                voteData.append({"vote": vote["SentimentValue"], "data": data["Text"] + "\n"})
                break
    for key in key_to_sentiment_mapping.keys():
        lines = [x["data"] for x in voteData if x["vote"] == key]
        filepath = os.path.join(dataset_path, key_to_sentiment_mapping[key], "{}.txt".format(key_to_sentiment_mapping[key]))
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        fo = codecs.open(filepath, 'w', encoding='utf-8')
        fo.writelines(lines)
        fo.close()