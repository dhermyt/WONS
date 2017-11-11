from nltk import NaiveBayesClassifier, SklearnClassifier
from sklearn.ensemble import RandomForestClassifier
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


class SklearnClassifierFactory(object):
    @staticmethod
    def SklearnMultinomialNB():
        return SklearnClassifierWrapper(MultinomialNB)

    @staticmethod
    def SklearnBernoulliNB():
        return SklearnClassifierWrapper(BernoulliNB)

    @staticmethod
    def SklearnLogisticRegression():
        return SklearnClassifierWrapper(LogisticRegression)

    @staticmethod
    def SklearnSGDClassifier():
        return SklearnClassifierWrapper(lambda: SGDClassifier(loss='log'))

    @staticmethod
    def SklearnSVC():
        return SklearnClassifierWrapper(lambda : SVC(probability=True))

    @staticmethod
    def SklearnLinearSVC():
        return SklearnClassifierWrapper(LinearSVC)

    @staticmethod
    def SklearnNuSVC():
        return SklearnClassifierWrapper(lambda : NuSVC(probability=True))

    @staticmethod
    def SklearnRidgeClassifier():
        return SklearnClassifierWrapper(RidgeClassifier)

    @staticmethod
    def SklearnPerceptron():
        return SklearnClassifierWrapper(Perceptron)

    @staticmethod
    def SklearnPassiveAggressive():
        return SklearnClassifierWrapper(PassiveAggressiveClassifier)

    @staticmethod
    def SklearnKNeighbours():
        return SklearnClassifierWrapper(KNeighborsClassifier)

    @staticmethod
    def SklearnNearestCentroid():
        return SklearnClassifierWrapper(NearestCentroid)

    @staticmethod
    def SklearnRandomForest():
        return SklearnClassifierWrapper(RandomForestClassifier)
