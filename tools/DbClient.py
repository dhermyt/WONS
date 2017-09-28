import pymongo
from urllib.parse import quote_plus
from datetime import datetime
class DbClient:
    __client = None
    __db = None
    __configuration = None

    def connect(self, configuration):

        uri = "mongodb://%s:%s@%s/wonsdb" % (
            quote_plus(configuration.DB_USER), quote_plus(configuration.DB_PASSWORD), "{}:{}".format(configuration.DB_HOST, configuration.DB_PORT))
        self.__client = pymongo.MongoClient(uri)
        self.__db = self.__client.wonsdb
        self.__configuration = configuration

    def get_user_id(self):
        userData = {"Username": self.__configuration.WONS_USERNAME}
        user = self.__db.Users.find_one(userData)
        if user is not None:
            return user["_id"]
        return self.__db.Users.insert_one(userData).inserted_id

    def get_user_votes(self, userid):
        cursor = self.__db.UserVotes.find({"UserId": userid})
        votes = []
        for vote in cursor:
            votes.append(vote)
        return votes

    def get_dataset(self):
        return self.__db.Dataset.find({"Source": self.__configuration.WONS_DATASET_SOURCE})

    def put_vote(self, userid, dataid, vote):
        userVote = {
            "UserId": userid,
            "TextDataId" :dataid,
            "SentimentValue": vote,
            "CreatedAt": datetime.now()
        }
        self.__db.UserVotes.insert_one(userVote)

    def delete_last_vote(self, userid):
        userVote = {
            "UserId": userid
        }
        vote = self.__db.UserVotes.find(userVote).sort("CreatedAt", pymongo.DESCENDING).next()
        self.__db.UserVotes.remove(vote)
