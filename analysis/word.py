from collections import OrderedDict

import nltk.data


def getSynonyms(word):
    synonyms = []
    for syn in nltk.corpus.wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.append(lemma.name())
    return list(OrderedDict.fromkeys(synonyms))
