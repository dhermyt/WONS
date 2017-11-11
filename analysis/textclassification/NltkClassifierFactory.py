from nltk import NaiveBayesClassifier, SklearnClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC

from analysis.textclassification.NltkClassifierWrapper import NltkClassifierWrapper
from analysis.textclassification.SklearnClassifierWrapper import SklearnClassifierWrapper


class NltkClassifierFactory(object):
    @staticmethod
    def SklearnMultinomialNB():
        return NltkClassifierWrapper(SklearnClassifier(MultinomialNB()))

    @staticmethod
    def SklearnBernoulliNB():
        return NltkClassifierWrapper(SklearnClassifier(BernoulliNB()))

    @staticmethod
    def SklearnLogisticRegression():
        return NltkClassifierWrapper(SklearnClassifier((LogisticRegression())))

    @staticmethod
    def SklearnSGDClassifier():
        return NltkClassifierWrapper(SklearnClassifier(SGDClassifier(loss='log')))

    @staticmethod
    def SklearnSVC():
        return NltkClassifierWrapper(SklearnClassifier(SVC(probability=True)))

    @staticmethod
    def SklearnLinearSVC():
        return NltkClassifierWrapper(SklearnClassifier(LinearSVC()))

    @staticmethod
    def SklearnNuSVC():
        return NltkClassifierWrapper(SklearnClassifier(NuSVC(probability=True)))

    @staticmethod
    def SklearnRidgeClassifier():
        return NltkClassifierWrapper(SklearnClassifier(RidgeClassifier()))

    @staticmethod
    def SklearnPerceptron():
        return NltkClassifierWrapper(SklearnClassifier(Perceptron()))

    @staticmethod
    def SklearnPassiveAggressive():
        return NltkClassifierWrapper(SklearnClassifier(PassiveAggressiveClassifier()))

    @staticmethod
    def SklearnKNeighbours():
        return NltkClassifierWrapper(SklearnClassifier(KNeighborsClassifier()))

    @staticmethod
    def SklearnNearestCentroid():
        return NltkClassifierWrapper(SklearnClassifier(NearestCentroid()))

    @staticmethod
    def SklearnRandomForest():
        return NltkClassifierWrapper(SklearnClassifier(RandomForestClassifier()))

    @staticmethod
    def SklearnVoting():
        return NltkClassifierWrapper(SklearnClassifier(VotingClassifier(
            estimators=[
                ('Perceptron', Perceptron()),
                ('PassiveAggressiveClassifier', PassiveAggressiveClassifier()),
                ('SGDClassifier', SGDClassifier(loss='log'))
            ])))
