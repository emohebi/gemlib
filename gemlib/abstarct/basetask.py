

class task():

    def __init__(self, task=None, task_name=None, cwd=None, title=None):
        self.definedtask = task
        self.task_name = task_name
        self.cwd = cwd
        self.title = title

class preprocessing():

    def __init__(self, preprocessing=None, preprocessing_name=None, preprocessing_parent_name=None):
        self.definedpreprocessing = preprocessing
        self.preprocessing_name = preprocessing_name
        self.preprocessing_parent_name = preprocessing_parent_name