from .utils.text import TextProcessor

_text_processor = TextProcessor()


class ProcessedExample:

    def __init__(self, premise, hypothesis, label):
        self._premise = premise
        self._hypothesis = hypothesis
        self._label = label

    @property
    def premise(self):
        return _text_processor.process(self._premise)

    @property
    def hypothesis(self):
        return _text_processor.process(self._hypothesis)

    @property
    def label(self):
        return self._label

    def __iter__(self):
        yield 'premise', self.premise
        yield 'hypothesis', self.hypothesis
        yield 'label', self.label
