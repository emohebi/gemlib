
# not in use yet
class output(object):
    def __init__(self, kwargs):
        self.scores = {}
        for k, v in kwargs.iteritems():
            if k == 'scores':
                self.scores = v