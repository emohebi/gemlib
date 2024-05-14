from gemlib.abstarct.basefunctionality import BaseTopicModelling
from pathlib import Path
from gensim.models.lsimodel import LsiModel
import pickle
from gensim.models import CoherenceModel
from tqdm import tqdm
import pandas as pd
from gemlib.validation import utilities

class LSITopicModelling(BaseTopicModelling):

    def get_model(self):
        self.file_label = f'{self.taskname}_{self.num_topics}'
        # Build LDA model
        utilities._info(f'building lsi model... per word topics: {str(self.perwordtopic)}')
        self.model = LsiModel(corpus=self.corpus,
                                  id2word=self.dictionary,
                                  num_topics=self.num_topics,
                                  # distributed=True,
                                  # update_every=1,
                                  chunksize=50000)  # ,
        # alpha='auto')
        # per_word_topics=True)
        #     pprint(lda_model.print_topics())
        if self.model:
            file = open(str(self.output_dir / Path(f'model_{self.file_label}.pkl')), 'wb')
            pickle.dump(self.model, file)
            file.close()
        else:
            raise ValueError