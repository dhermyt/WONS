import json


class SettingsEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, type):
            return obj.__name__
        if callable(obj):
            return obj.__name__
        return json.JSONEncoder.default(self, obj)
