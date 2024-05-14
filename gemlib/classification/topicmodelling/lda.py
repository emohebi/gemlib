from gemlib.abstarct.basefunctionality import BaseTopicModelling
from pathlib import Path
from gensim.models.ldamulticore import LdaMulticore
import pickle
from gensim.models import CoherenceModel
from tqdm import tqdm
import pandas as pd
from gemlib.validation import utilities

class LDATopicModelling(BaseTopicModelling):

    def get_model(self):
        self.file_label = f'{self.taskname}_{self.num_topics}'
        # Build LDA model
        utilities._info(f'building lda model... per word topics: {str(self.perwordtopic)}')
        self.model = LdaMulticore(corpus=self.corpus,
                                   id2word=self.dictionary,
                                   num_topics=self.num_topics,
                                   random_state=100,
                                   workers=5,
                                   # update_every=1,
                                   chunksize=100000,
                                   passes=5,
                                   iterations=100,
                                   per_word_topics=self.perwordtopic)#,
                                   # alpha='auto')
                                   # per_word_topics=True)
    #     pprint(lda_model.print_topics())
        if self.model:
            file = open(str(self.output_dir / Path(f'model_{self.file_label}.pkl')), 'wb')
            pickle.dump(self.model, file)
            file.close()
        else:
            raise ValueError