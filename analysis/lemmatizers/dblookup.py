import sqlite3

from definitions import POLIMORF_DB_PATH


class DbLookupLemmatizer(object):
    __conn = None
    __c = None
    __dbpath = POLIMORF_DB_PATH

    @property
    def is_initialized(self):
        return self.__conn is not None

    def initialize(self):
        self.__conn = sqlite3.connect(self.__dbpath)
        self.__c = self.__conn.cursor()

    def get_lemma(self, lexeme):
        if self.is_initialized is False:
            return None
        self.__c.execute('SELECT {cn} FROM {tn} WHERE {id}=?'.format(tn='Lemmas', cn='Lemma', id='Lexeme'), (lexeme,))
        lemma = self.__c.fetchone()
        return lemma[0] if lemma is not None else lexeme

    def close(self):
        self.__conn.close()
