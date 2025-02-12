import json

def load_candidates(filename=None):
    with open(filename) as fp:
        return json.load(fp)
