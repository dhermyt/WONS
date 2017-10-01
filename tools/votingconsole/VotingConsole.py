import os
import msvcrt

from tools.votingconsole.toolsettings import ToolSettings
from tools.DbClient import DbClient
from tools.votingconsole.TextClassifier import TextClassifier


def cls():
    os.system('cls')


key_to_sentiment_mapping = {
    '0': '2',
    '1': '-1',
    '2': '0',
    '3': '1'
}


def has_already_voted(textData, user, userVotes):
    vote = next((vote for vote in userVotes if vote["UserId"] == user and vote["TextDataId"] == textData["_id"]), None)
    return vote is not None


if __name__ == '__main__':
    client = DbClient()
    toolSettings = ToolSettings()
    client.connect(toolSettings.DB_USER, toolSettings.DB_PASSWORD, toolSettings.DB_HOST, toolSettings.DB_PORT)
    user = client.get_user_id(toolSettings.WONS_USERNAME)
    userVotes = client.get_user_votes(user)
    dataset = client.get_dataset(toolSettings.WONS_DATASET_SOURCE)
    classifier = TextClassifier()
    classifier.initialize()
    countToVote = 0
    for textData in dataset:
        if not has_already_voted(textData, user, userVotes) and classifier.matches(textData["Text"]):
            countToVote += 1
    print("Text data to vote: {}".format(countToVote))
    msvcrt.getch().decode('ASCII')
    dataset = client.get_dataset()
    cls()
    for textData in dataset:
        if has_already_voted(textData, user, userVotes):
            continue
        if not classifier.matches(textData["Text"]):
            continue

        print()
        print(textData["Text"])
        vote = msvcrt.getch().decode('ASCII')
        if vote == ",":
            client.delete_last_vote(user)
            continue
        client.put_vote(user, textData["_id"], key_to_sentiment_mapping[vote])
        cls()
        print("Already voted: {}".format(len(userVotes)))
