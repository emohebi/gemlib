from gemlib.abstarct.basefunctionality import BaseFeatureEngineering
from gemlib.featureengineering.featureengineer import text_encoding
import numpy as np
import pandas as pd
from gemlib.validation import utilities
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from multiprocessing import Pool
from pathlib import Path

class textextraction(BaseFeatureEngineering):

    def __init__(self, name: str = None, definition: dict = None, save: bool = False, dirpath: str = None, groupby: list = None, condition: dict = None, keepcolumnsvalues: list = None, axis: int = None, num_vocab: int = None, embedding_dim: int = None, embedding_path: str = None, input: list = None, columns: list = None, cache: bool = False, concat: bool = False, x_cols: list = None, y_col: str = None, test_ratio: float = None, tokenizer=None, column: str = None, sequence_lenght: int = None, chunk_size: int = None, models: dict = None, text_col: str = None, id_col: str = None, terms_col: str = None, num_cores: int = None, df_aux: pd.DataFrame = None):
        super().__init__(name, definition, save, dirpath, groupby, condition, keepcolumnsvalues, axis, num_vocab, embedding_dim, embedding_path, input, columns, cache, concat, x_cols, y_col, test_ratio, tokenizer, column, sequence_lenght, chunk_size, models, text_col, id_col, terms_col, num_cores, df_aux)
        try:
            import en_core_web_lg
        except:
            utilities._info('downloading spacy model: en_core_web_lg')
            with utilities.Spinner():  
                from spacy.cli import download
                download('en_core_web_lg')
        try:
            import en_core_web_lg
            self.nlp = en_core_web_lg.load()
        except:
            import spacy
            self.nlp = spacy.load('en_core_web_lg')

    def get_pos_patterns(self, text):
        patterns = []
        matcher = utilities.get_pos_matcher(self.nlp)
        doc = self.nlp(text)
        matches = matcher(doc)
        patterns.extend([doc[start:end].text for _, start, end in matches])
        return list(set(patterns))

    def get_overlapped_text(self, expression_list:list):
        overl = []
        for i in range(len(expression_list)):
            for j in range(i+1, len(expression_list)):
                if expression_list[i] in expression_list[j]:
                    overl.append(expression_list[i])
        return overl

    def extract_patterns(self, args):
        text, text_col, id_col , is_overlap_detector = args

        patterns = self.get_pos_patterns(text.lower())
        if text_col in patterns:
            patterns.remove(text_col)
        patterns = np.array(patterns).reshape(-1,1)
        patterns = np.unique(patterns)
        if patterns.shape[0] < 1:
            return [[id_col, '']]
        if is_overlap_detector:
            patterns = patterns.tolist()
            patterns_tbd = self.get_overlapped_text(patterns)
            patterns = np.array(list(set(patterns) - set(patterns_tbd)))

        all_res = np.dstack([np.repeat(id_col, patterns.shape[0]), patterns])[0].tolist()

        return all_res

    def get_prediction(self, args):
        terms, terms_col, df_expressions, terms_embeddings, expressions_embeddings, pickled_dir, file_label= args
        lent = len(terms_embeddings)
        coef = 1/lent
        similarities = None
        for i in range(lent):
            if similarities is None:
                similarities = coef * cosine_similarity(expressions_embeddings[i], terms_embeddings[i])
            else:
                similarities += coef * cosine_similarity(expressions_embeddings[i], terms_embeddings[i])

        df = pd.DataFrame(similarities.argsort()[:,-1:]).T.unstack().to_frame().reset_index().rename({'level_0':'index', 
                                                                                                    'level_1':'rank',
                                                                                                    0:terms_col}, axis='columns')
        df_p = pd.DataFrame(np.sort(similarities)[:,-1:]).T.unstack().to_frame().reset_index().rename({'level_0':'index', 
                                                                                                    'level_1':'rank',
                                                                                                    0:'similarity'}, axis='columns')
        df = df.merge(df_p, on=['index', 'rank'])
        df.drop(['index', 'rank'], axis='columns', inplace=True)
        df.set_index(df_expressions.index, inplace=True)
        df = df.join(df_expressions)
        _mapping = {c:terms[c] for c in range(len(terms))}
        df[terms_col] = df[terms_col].map(_mapping)
        df[df.similarity>=self.threshold].to_csv(pickled_dir /f'preds_{self.name}_{file_label}.csv', index=False)

    def apply(self, df=None):
        output_col = 'expressions'

        if self.is_encode_terms:
            txt_encode = text_encoding(self.terms_col, self.chunk_size, models=self.models, device=self.device)
            encoded_terms = txt_encode.apply(self.df_aux)
            utilities._info('terms encoding is done...')
            resources = utilities.resolve_caching_stage(self.cache, encoded_terms, self.dirpath, self.sub_folder, f'_terms')
            self.resources.update(resources)
            utilities.stage_resources_dict(self.resources, self.dirpath)
        
        
        if self.is_run_text_extractor:
            if (Path(self.dirpath) / self.sub_folder / f'{self.name}_expressions_0.csv').is_file():
                return 
            args = []
            num_cores = self.num_cores
            for text, id_ in df[[self.text_col, self.id_col]].values.tolist():
                args.append([text.lower(), self.terms_col, id_ , self.is_overlap_detection])

            with Pool(num_cores) as p:
                results_list = list(tqdm(p.imap(self.extract_patterns, args), total=len(args)))

            df_ = pd.DataFrame(np.concatenate(results_list), columns=[self.id_col, output_col])
            resources = utilities.resolve_caching_stage(self.cache, df_, self.dirpath, self.sub_folder, f'{self.name}_expressions')
            self.resources.update(resources)
            utilities.stage_resources_dict(self.resources, self.dirpath)

        if self.is_encode_expressions:
            df_ = utilities.resolve_caching_load(self.resources, f'{self.name}_expressions')
            df_.dropna(inplace=True)
            df_[output_col] = df_[output_col].astype(str)
            txt_encode = text_encoding(output_col, self.chunk_size, models=self.models, device=self.device)
            encoded_expr = txt_encode.apply(df_)
            utilities._info('expressions encoding is done...')
            resources = utilities.resolve_caching_stage(self.cache, encoded_expr, self.dirpath, self.sub_folder, f'{self.name}_expressions')
            self.resources.update(resources)
            utilities.stage_resources_dict(self.resources, self.dirpath, self.name)
        
        if self.is_run_similarity_check:
            n = self.chunk_size
            terms_embeddings = []
            for m in self.models:
                terms_embeddings.append(utilities.resolve_caching_load(self.resources, 
                                        input=f'_terms_0_model_{m}_encode', 
                                        concat=True))
            df_ = utilities.resolve_caching_load(self.resources, f'{self.name}_expressions')
            df_.dropna(inplace=True)
            df_[output_col] = df_[output_col].astype(str)
            all_res = []
            for index, j in tqdm(enumerate(range(0, df_[output_col].shape[0], n))):
                exprssions_embeddings = []
                for m in self.models:
                    exprssions_embeddings.append(pickle.load(open(self.resources[f'{self.name}_expressions_0_model_{m}_encode'][index], 
                                                                                'rb')))
                df_tmp = df_[j:j + n].reset_index(drop=True)
                m = self.inner_chunk_size
                args = []
                num_cores = self.num_cores
                for ind, i in enumerate(range(0, df_tmp[output_col].shape[0], m)):
                    df = df_tmp[i:i + m].drop_duplicates([self.id_col, output_col])
                    args.append([self.df_aux[self.terms_col].values.tolist(),
                                self.terms_col, 
                                df,
                                terms_embeddings, 
                                [embed[df.index.tolist()] for embed in exprssions_embeddings],
                                Path(self.dirpath) / self.sub_folder,
                                f'{index}_{ind}'])
                

                with Pool(num_cores) as p:
                    results_list = list(tqdm(p.imap(self.get_prediction, args), total=len(args)))

            utilities._info('all extrctions done! .. ')

            all_res = []
            for f in tqdm(list((Path(self.dirpath) / self.sub_folder).glob(f'preds_{self.name}_*.csv'))):
                all_res.append(pd.read_csv(f))

            df_ = pd.concat(all_res)
            df_.reset_index(drop=True, inplace=True)
            df_['len_ext'] = [len(kw.split(' ')) for kw in df_[output_col].values.tolist()]
            resources = utilities.resolve_caching_stage(self.cache, df_, self.dirpath, 'results', f'{self.name}_output')
