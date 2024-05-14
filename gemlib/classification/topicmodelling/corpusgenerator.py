import re
import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm
import os
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gemlib.validation import utilities
from gemlib.abstarct.basefunctionality import BaseTopicModelling
# spacy for lemmatization
#import spacy

# Plotting tools
#import pyLDAvis
#import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
directory = r"D:\Projects\SocialMediaEventDetection"
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import ast
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
from nltk.corpus import wordnet
wn_lemmas = set(wordnet.all_lemma_names())
#nlp = spacy.load('en', disable=['parser', 'ner'])
import string
ascii_chars = set(string.printable)  # speeds things up
import pickle
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.tfidfmodel import TfidfModel
from pathlib import Path
from collections import defaultdict

class Corpus(BaseTopicModelling):

    def preprocess_data(self, data):
        data = [" ".join(sent.splitlines()) for sent in data]

        highpoints = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
        pre = re.compile(r'[^\w\s]', re.UNICODE)
        stoplist = [u'merc\xe8', u'lamerc\xe8']
        stoplist.extend(stopwords.words('english'))

        # remove urls
        data = [re.sub(
            r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+'
            r'|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))'
            r'+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
            '', hit) for hit in data]
        # remove emojii
        data = [highpoints.sub('', tweet) for tweet in data]

        data = [re.sub(',','', tweet) for tweet in data]
        # remove numbers
        data = [re.sub("\d+", "", tweet) for tweet in data]
        # remove mentions and the hash sign
        data = [re.sub("@", "", tweet).lower() for tweet in data]
        data = [re.sub("(@[A-Za-z0-9]+)", "", tweet).lower() for tweet in data]
        data = [re.sub("#", "", tweet) for tweet in data]
        data = [re.sub("_", "", tweet) for tweet in data]
        data = [pre.sub("", tweet) for tweet in data]
        # Remove Emails
        data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

        data = [re.sub('amp', '', sent) for sent in data]
        # Remove new line characters
        data = [re.sub('\s+', ' ', sent) for sent in data]
        # Remove distracting single quotes
        data = [re.sub("\'", "", sent) for sent in data]

        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        data = [emoji_pattern.sub(r'', text) for text in data]
        # pprint(data[:10])
        return data

    def remove_low_freq_words(self, texts):
        word_freq = defaultdict(int)
        for sent in texts:
            if len(sent) < 1: continue
            for i in sent:
                if len(i) < 1: continue
                word_freq[i] += 1
        df = pd.DataFrame([[m, word_freq[m]] for m in word_freq],columns=['term', 'freq'])
        # df = df[df['freq'] > 300]
        terms = set(df.term.values.tolist())
        texts = [[word for word in doc if word in terms] for doc in texts]
        texts_len = np.array([len(sent) for sent in texts])
        mask_low_freq = texts_len > 1
        texts = [sent for sent in texts if len(sent) > 1]
        return texts, mask_low_freq

    def remove_non_ascii_prinatble_from_list(self, texts):
        return [[word for word in doc if all(char in ascii_chars for char in word)] for doc in texts]

    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield (simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(self, texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def remove_non_english_words(self, texts):
        return [[word for word in simple_preprocess(str(doc)) if word in wn_lemmas] for doc in texts]

    def make_bigrams(self, texts, bigram_mod):
        return [bigram_mod[doc] for doc in texts]


    def make_trigrams(self, texts, bigram_mod, trigram_mod):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]


    def lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        #for sent in texts:
        #    doc = nlp(" ".join(sent))
        #    texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def get_preprocessed_data(self, data):
        data = self.preprocess_data(data)
        data_words = list(self.sent_to_words(data))

        # utilities._info(data_words[:1])
        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=10)  # higher threshold fewer phrases.
        # trigram = gensim.models.Phrases(bigram[data_words], threshold=10)

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        # trigram_mod = gensim.models.phrases.Phraser(trigram)

        # See trigram example
        # utilities._info(trigram_mod[bigram_mod[data_words[0]]])

        # Remove Stop Words
        data_words_nostops = self.remove_stopwords(data_words)

        # remove non english char words
        data_words_nostops = self.remove_non_ascii_prinatble_from_list(data_words_nostops)

        data_words_nostops = self.remove_non_english_words(data_words_nostops)

        if self.use_phrases:
            utilities._info('extracting phrases (bigrams)...')
            # Form Bigrams
            data_words_nostops = self.make_bigrams(data_words_nostops, bigram_mod)  # , trigram_mod)

        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = self.lemmatization(data_words_nostops, allowed_postags=['NOUN'])  # 'ADJ', , 'ADV', 'VERB'
        # utilities._info(data_lemmatized[:1])

        # Remove Stop Words again
        data_lemmatized = self.remove_stopwords(data_lemmatized)

        utilities._info(f'len data before filtering: {len(data_lemmatized)}')
        mask_low_freq = None
        # data_lemmatized, mask_low_freq = self.remove_low_freq_words(data_lemmatized)

        utilities._info(f'len data after filtering: {len(data_lemmatized)}')
        return data_lemmatized, mask_low_freq

    def get_corpus(self, data=None):

        data_lemmatized, mask_low_freq = self.get_preprocessed_data(data)
        # Create Dictionary
        dictionary = corpora.Dictionary(data_lemmatized)

        # Filter the terms which have occured in less than 3 articles and more than 40% of the articles
        # if not self.allowext:
        #     dictionary.filter_extremes(no_below=4, no_above=0.4)

        # List of some words which has to be removed from dictionary as they are content neutral words
        stoplist = set('Fuck fuck shit hahaha haha hahahaha OMG omg'.split())
        stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
        dictionary.filter_tokens(stop_ids)

        # Create Corpus
        try:
            cfile = open(str(self.output_dir / Path(f'data_lemmatized_{self.taskname}.pkl')), 'wb')
            pickle.dump(data_lemmatized, cfile)
            cfile.close()
            utilities._info('data_lemmatized pickled...')
        except :
            utilities._info('data_lemmatized dumping is failed .. it is ignored...')
            pass

        try:
            cfile = open(str(self.output_dir / Path(f'dictionary_{self.taskname}.pkl')), 'wb')
            pickle.dump(dictionary, cfile)
            cfile.close()
            utilities._info('dictionary pickled...')
        except :
            utilities._info('dictionary dumping is failed .. it is ignored...')
            pass

        # Term Document Frequency
        corpus = [dictionary.doc2bow(text) for text in data_lemmatized]

        if self.use_tfidf:
            utilities._info('tf/idf modelling...')
            tfidf = TfidfModel(corpus)
            corpus = tfidf[corpus]
        try:
            cfile = open(str(self.output_dir / Path(f'corpus_{self.taskname}.pkl')), 'wb')
            pickle.dump(corpus, cfile)
            cfile.close()
            utilities._info('corpus pickled...')
        except :
            utilities._info('corpus dumping is failed .. it is ignored...')
            pass
        # View
        utilities._info(f'len of corpus: {len(corpus)} ...')
        if len(corpus) < self.min_num_docs:
            return None
        return corpus, dictionary, mask_low_freq

    def get_model(self):
        pass

    def run(self):
        pass

    def output_topics(self, _df):
        pass
