from gemlib.classification.topicmodelling.tweets.preprocess import TwitterPreprocessing
from gemlib.validation import utilities
import sys

class TextPreprocessingValidation(object):
    def __init__(self, object):
        self.validate_instance = object

    def initialize(self):
        bins = 10

        if not 'type' in self.validate_instance:
            utilities._error(0, '"type" of preprocessing must be set.')

        if 'bins' in self.validate_instance:
            bins = float(self.validate_instance['bins'])

        if self.validate_instance['type'].lower() == 'twitter':
            return TwitterPreprocessing(bins=bins)
        else:
            raise NotImplementedError

    def get_valid_textpreprocessing(self):
        try:
            textpreprocess = self.initialize()
            if textpreprocess is None:
                utilities._info("not a valid textpreprocessing !!! ...")
            return textpreprocess
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("textpreprocessing initialization failed!!!\n {0}".format(exceptionValue))


