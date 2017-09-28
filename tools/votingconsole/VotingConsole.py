import os
import msvcrt

from tools.votingconsole.toolsettings import Settings as ToolSettings
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
    os.environ["DEBUG"] = "1"
    client = DbClient()
    client.connect(ToolSettings())
    user = client.get_user_id()
    userVotes = client.get_user_votes(user)
    dataset = client.get_dataset()
    classifier = TextClassifier()
    classifier.initialize()
    cls()
    for textData in dataset:
        if has_already_voted(textData, user, userVotes):
            continue
        if not classifier.matches(textData["Text"]):
            continue

        print("Already voted: {}".format(len(userVotes)))
        print()
        print(textData["Text"])
        vote = msvcrt.getch().decode('ASCII')
        if vote == ",":
            client.delete_last_vote(user)
            continue
        client.put_vote(user, textData["_id"], key_to_sentiment_mapping[vote])
        cls()
