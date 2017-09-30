import sys, os

from analysis.textclassification import bagofwords
from configuration.Decoder import SettingsDecoder
from tools.SentimentAnalysisToolbox import SentimentAnalysisToolbox

sys.path.append('D:\\home\\site\\wwwroot\\env\\Lib\\site-packages')

from voluptuous import Required, Schema, All, Length, MultipleInvalid, In, Optional

from configuration.appsettings import Settings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, redirect
from flask_restful import Resource, Api, reqparse
from applicationinsights.requests import WSGIApplication

app = Flask(__name__, static_folder='assets')
if os.environ.get('APPSETTING_APPLICATION_INSIGHTS_INSTRUMENTATION_KEY') is not None:
    app.wsgi_app = WSGIApplication(os.environ.get('APPSETTING_APPLICATION_INSIGHTS_INSTRUMENTATION_KEY'), app.wsgi_app)
api = Api(app)


class SentimentAnalysis(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('text')
    schema = Schema({
        Required('text'): All(str, Length(min=1, max=1000))
    })
    toolbox = SentimentAnalysisToolbox()
    decoder = SettingsDecoder()
    settings = toolbox.load_json('default', Settings())
    settings.LEMMATIZER_TYPE = decoder.decode_type(settings.LEMMATIZER_TYPE)
    settings.CLASSIFIER_TYPE = decoder.decode_classifier(settings.CLASSIFIER_TYPE)
    classifier = settings.CLASSIFIER_TYPE()
    classifier.load('default')

    def post(self):
        args = self.parser.parse_args()
        try:
            self.schema(args)
        except MultipleInvalid as e:
            return str(e), 400, {'Access-Control-Allow-Origin': '*'}
        text = bagofwords.get_processed_bag_of_words(args['text'], None, self.settings)
        sa = self.classifier.classify(text)
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


api.add_resource(SentimentAnalysis, '/api/text/sentimentanalysis')

if __name__ == '__main__':
    app.run()
