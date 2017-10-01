import pymongo
from urllib.parse import quote_plus
from datetime import datetime

class DbClient:
    __client = None
    __db = None

    def connect(self, user, password, host ,port):

        uri = "mongodb://%s:%s@%s/wonsdb" % (
            quote_plus(user), quote_plus(password), "{}:{}".format(host, port))
        self.__client = pymongo.MongoClient(uri)
        self.__db = self.__client.wonsdb

    def get_user_id(self, username):
        userData = {"Username": username}
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

    def get_dataset(self, dataset):
        return self.__db.Dataset.find({"Source": dataset})

    def put_vote(self, userid, dataid, vote):
        userVote = {
            "UserId": userid,
            "TextDataId" :dataid,
            "SentimentValue": vote,
            "CreatedAt": datetime.utcnow()
        }
        self.__db.UserVotes.insert_one(userVote)

    def delete_last_vote(self, userid):
        userVote = {
            "UserId": userid
        }
        vote = self.__db.UserVotes.find(userVote).sort("CreatedAt", pymongo.DESCENDING).next()
        self.__db.UserVotes.remove(vote)
