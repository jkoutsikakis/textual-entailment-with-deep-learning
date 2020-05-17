import spacy


class TextProcessor:
    def __init__(self):
        self._spacy_nlp = spacy.load('en', disable=['parser', 'ner'])

    def process(self, text):
        if not text:
            return []
        return [t.text.lower() for t in self._spacy_nlp(text)]
