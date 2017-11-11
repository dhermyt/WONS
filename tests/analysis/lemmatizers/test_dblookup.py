import unittest
from analysis.lemmatizers.dblookup import DbLookupLemmatizer


class Test_DbLookup(unittest.TestCase):
    def test_get_lemma(self):
        dblookup = DbLookupLemmatizer()
        dblookup.initialize()
        lemma = dblookup.get_lemma('otrzymała')
        self.assertTrue(lemma == 'otrzymać')
        dblookup.close()

    def test_get_unknown_lemma(self):
        lexeme = 'abcdefgh'
        dblookup = DbLookupLemmatizer()
        dblookup.initialize()
        lemma = dblookup.get_lemma(lexeme)
        self.assertTrue(lemma == lexeme)
        dblookup.close()


if __name__ == '__main__':
    unittest.main()
