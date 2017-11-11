import sys, os

from analysis.textclassification import bagofwords
from analysis.textclassification.NltkClassifierFactory import NltkClassifierFactory

sys.path.append('D:\\home\\site\\wwwroot\\env\\Lib\\site-packages')

from voluptuous import Required, Schema, All, Length, MultipleInvalid, In, Optional

from configuration.appsettings import Settings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, redirect
from flask_restful import Resource, Api, reqparse
import analysis.word
from applicationinsights.requests import WSGIApplication

app = Flask(__name__, static_folder='assets')
if os.environ.get('APPSETTING_APPLICATION_INSIGHTS_INSTRUMENTATION_KEY') is not None:
    app.wsgi_app = WSGIApplication(os.environ.get('APPSETTING_APPLICATION_INSIGHTS_INSTRUMENTATION_KEY'), app.wsgi_app)
api = Api(app)


class WordSynonyms(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('word')
    schema = Schema({
        Required('word'): All(str, Length(min=1, max=200))
    })

    def post(self):
        args = self.parser.parse_args()
        try:
            self.schema(args)
        except MultipleInvalid as e:
            return str(e), 400, {'Access-Control-Allow-Origin': '*'}
        synonyms = analysis.word.getSynonyms(args['word'])
        return {'word': args['word'], 'synonyms': synonyms}, 200, {'Access-Control-Allow-Origin': '*'}

    def options(self):
        return {'Allow': 'POST'}, 200, \
               {'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST'}


class SentimentAnalysis(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('text')
    parser.add_argument('trainingset')
    classifiers = {'product_reviews': NltkClassifierFactory.SklearnMultinomialNB(),
                   'people_opinions': NltkClassifierFactory.SklearnVoting()}
    schema = Schema({
        Required('text'): All(str, Length(min=1, max=1000)),
        Optional('trainingset'): In(list(classifiers.keys()))
    })
    for key, value in classifiers.items():
        value.load(key)
    s = Settings()
    s.filterStopwords = True
    s.maxNgrams = 3

    def post(self):
        args = self.parser.parse_args()
        try:
            self.schema(args)
        except MultipleInvalid as e:
            return str(e), 400, {'Access-Control-Allow-Origin': '*'}
        text = bagofwords.get_processed_bag_of_words(args['text'], None, self.s)
        sa = self.classifiers[args['trainingset']].classify(text, 'neutral')
        return {'text': args['text'], 'sentiment': sa}, 200, {'Access-Control-Allow-Origin': '*'}


@app.route('/')
def index():
    return redirect("http://www.wons.net.pl", code=302)


@app.before_first_request
def initialize():
    import logging
    fileHandler = logging.FileHandler('api_logs.txt')
    fileHandler.setLevel(logging.DEBUG)
    app.logger.setLevel(logging.DEBUG)
    app.logger.addHandler(fileHandler)


@app.after_request
def after_request(response):
    app.logger.debug('response_body: %s', response.get_data())
    return response


api.add_resource(WordSynonyms, '/api/word/synonyms')
api.add_resource(SentimentAnalysis, '/api/text/sentimentanalysis')

if __name__ == '__main__':
    app.run()
