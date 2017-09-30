import json
from pydoc import locate

from analysis.lemmatizers.dblookup import DbLookupLemmatizer
from analysis.textclassification.NltkClassifierFactory import NltkClassifierFactory

known_types = {
    'DbLookupLemmatizer' : DbLookupLemmatizer
}

class SettingsDecoder(json.JSONDecoder):
    def object_hook(self, obj):
        if obj.startswith("type:"):
            typeName = obj.split(":")[1]
            return locate(typeName)
        return obj

    def decode_type(self, str):
        return known_types[str]

    def decode_classifier(self, str):
        return getattr(NltkClassifierFactory, str)

