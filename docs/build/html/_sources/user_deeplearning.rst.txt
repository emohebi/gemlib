Deep Learning Classification 
==================================

The deep learnig models in ``gemlib`` library can be called using a pre-defined names. The first model is called ``dl_ff_lstm_cnn``.

``dl_ff_lstm_cnn`` model
-------------------------
The model is a hybrid deep learning model consist of feed-forward, cnn and lstm layers. The architecture of the model is presented in 
below figure.

.. figure:: ./images/dl_ff_lstm_cnn.png
    :scale: 35%
    :align: center
    :alt: alternate text
    :figclass: align-center


The deep learning classification in the ``classification`` class can be called based on two scenarios, `using a pipeline` or 
`outside a pipeline`.


Using a pipeline
*****************

The models can be trained/validated by calling the `gemlib` cmd using a pipeline (json/yaml) file as an input:

.. code-block:: bash

    $ gemlib -i <path to input json file>

Example of input pipeline for training a ``dl_ff_lstm_cnn`` model:

.. code-block:: json

    {
        "cwd": "C:/Users/EM2945/Documents/CWD/test_gemlib_1k",
        "data": {
            "BG1": {
                "type": "csv",
                "path": "C:/Users/em2945/Documents/Data/bg_flow_18.01.2021_jobtitles_min.csv",
                "drop_na": ["TITLE", "ANZSCO_CODE"],
                "num_rows": 1000,
                "preprocessing": {
                    "P1":{
                        "input": "BG1",
                        "process": "featureengineering",
                        "type": "text_cleaning",
                        "columns": ["TITLE"]
                    },
                    "P2":{
                        "input": "BG1_P1_0",
                        "process": "featureengineering",
                        "type": "train_test_split",
                        "x_cols": ["TITLE"],
                        "y_col": "ANZSCO_CODE",
                        "test_ratio": 0.2
                    },
                    "P3":{
                        "input": ["BG1_P2_y_train", "BG1_P2_y_test"],
                        "process": "featureengineering",
                        "type": "factorization"
                    }
                }
            }
        },
        "preprocessing": {
            "P1":{
                "input": ["BG1_P2_x_train", "BG1_P2_x_test"],
                "process": "featureengineering",
                "type": "text_tokenization",
                "sequence_length": 10
            },
            "P2":{
                "input": ["BG1_P2_x_train", "BG1_P2_x_test"],
                "process": "featureengineering",
                "type": "text_encoding",
                "chunk_size": 300
            }
        },
        "tasks": {
            "T1": {
                "cwd": "C:/Users/EM2945/Documents/CWD/test_gemlib_1k/dl_classification",
                "type": "classification",
                "algorithm": "deeplearning",
                "model_name": "dl_ff_lstm_cnn",
                "mode": "training",
                "x_train": ["P1_0", "P1_0", "P2_0_model_1_encode", "P2_0_model_3_encode"],
                "y_train": "BG1_P3_0",
                "x_test": ["P1_1", "P1_1", "P2_1_model_1_encode", "P2_1_model_3_encode"],
                "y_test": "BG1_P3_1",
                "embedding_matrix": "T1_embedding_1_0",
                "epoch": 1000,
                "batch_size": 512,
                "sequence_length": 10,       
                "preprocessing": {
                    "embedding_1": {
                        "process": "featureengineering",
                        "type": "embedding",
                        "embedding_dim": 1024,
                        "embedding_path": "C:/Users/em2945/Documents/JobTitle_to_ANZSCO_Classification/pickled/x_train_word_embeddings.pkl",
                        "num_vocabs": 400000,
                        "tokenizer": "P1_tokenizer"
                    }
                }
            }
        }
    }

.. note:: 

    All the intermediate artefacts will be staged in ``cwd``. A ``resources.json`` file as key-value pair will be generated, 
    where ``key`` is unique based on the keys in input file and values are the uri to the artefacts.
    An Example of output ``resources.json`` file:

    .. code-block:: json

        {
            "BG1": [
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\data\\BG1_0.csv"
            ],
            "BG1_P1_0": [
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\BG1_P1_0_0.csv"
            ],
            "BG1_P2_x_train": [
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\BG1_P2_x_train_0.pkl"
            ],
            "BG1_P2_x_test": [
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\BG1_P2_x_test_0.pkl"
            ],
            "BG1_P2_y_train": [
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\BG1_P2_y_train_0.pkl"
            ],
            "BG1_P2_y_test": [
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\BG1_P2_y_test_0.pkl"
            ],
            "BG1_P3_0": [
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\BG1_P3_0_0.pkl"
            ],
            "BG1_P3_1": [
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\BG1_P3_1_0.pkl"
            ],
            "BG1_P3_factor_mapping": [
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\BG1_P3_factor_mapping_0.pkl"
            ],
            "P1_0": [
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P1_0_0.pkl"
            ],
            "P1_1": [
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P1_1_0.pkl"
            ],
            "P1_tokenizer": [
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P1_tokenizer_0.pkl"
            ],
            "P2_0_model_1_encode": [
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P2_0_model_1_encode_0.pkl",
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P2_0_model_1_encode_1.pkl",
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P2_0_model_1_encode_2.pkl",
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P2_0_model_1_encode_3.pkl",
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P2_0_model_1_encode_4.pkl",
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P2_0_model_1_encode_5.pkl",
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P2_0_model_1_encode_6.pkl"
            ],
            "P2_1_model_1_encode": [
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P2_1_model_1_encode_0.pkl",
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P2_1_model_1_encode_1.pkl"
            ],
            "P2_0_model_3_encode": [
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P2_0_model_3_encode_0.pkl",
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P2_0_model_3_encode_1.pkl",
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P2_0_model_3_encode_2.pkl",
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P2_0_model_3_encode_3.pkl",
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P2_0_model_3_encode_4.pkl",
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P2_0_model_3_encode_5.pkl",
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P2_0_model_3_encode_6.pkl"
            ],
            "P2_1_model_3_encode": [
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P2_1_model_3_encode_0.pkl",
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\P2_1_model_3_encode_1.pkl"
            ],
            "T1_embedding_1_0": [
                "C:\\Users\\EM2945\\Documents\\CWD\\algo_run_trans\\preprocessing\\T1_embedding_1_0_0.pkl"
            ]
        }


Outside a pipeline
*******************

Deep learning classification can be intialized and run outside a pipeline. The input data should be prepared before 
passing it into the training fucntion.

Example of training a ``dl_ff_lstm_cnn`` model:

.. code-block:: python

    from gemlib.validation import utilities
    from gemlib.classification.deeplearning import DeepLearning
    from gemlib.featureengineering.featureengineer import text_tokenization, text_encoding 
    from gemlib.featureengineering.featureengineer import train_test_splitting, factorization

    # get train data from df (pandas dataframe)        
    train_data = train_test_splitting(x_cols=x_col, y_col=y_col, test_ratio=0.2)
    data_dict = train_data.apply(df)

    # text tokenization
    text_token = text_tokenization(column=x_col, sequence_lenght=10)
    token_dict = text_token.apply([data_dict['x_train'][0], data_dict['x_test'][0]])

    # text encoding using bert models
    text_encode = text_encoding(column=x_col, chunk_size=100, device='cpu')
    data_encode = text_encode.apply([data_dict['x_train'][0], data_dict['x_test'][0]])

    # target factorization
    factorize = factorization()
    data_y = factorize.apply([data_dict['y_train'][0], data_dict['y_test'][0]])

    x_train = [token_dict['0'][0], 
               token_dict['0'][0], 
               np.concatenate(data_encode['0_model_1_encode'], axis=0),
               np.concatenate(data_encode['0_model_3_encode'], axis=0)]

    x_test = [token_dict['1'][0], 
              token_dict['1'][0], 
              np.concatenate(data_encode['1_model_1_encode'], axis=0),
              np.concatenate(data_encode['1_model_3_encode'], axis=0)]

    y_train = data_y['0']
    y_test = data_y['1']

    # initializes the deep learnig model
    dl = DeepLearning(x_train=x_train, x_test=x_test, 
                    y_train=y_train, y_test=y_test, 
                    sequence_lenght=10, embedding_matrix='T1_embedding_1_0',
                    model_name='dl_ff_lstm_cnn',
                    epoch=100, batch_size=32)
    dl.dirpath = r"C:\Users\em2945\Documents\CWD\test_gemlib"
    dl.get_model() # initializes the model 'dl_ff_lstm_cnn'
    dl.run_training() # train the model

The validation can be done in the same way if we have trained a model beforehand. The sample code block presented below
validate the model against any dataset.

.. code-block:: python

    # loading the dictioanry of resources 
    resources_path = r"C:\Users\em2945\Documents\CWD\test_gemlib\resources.json"
    with open(resources_path, 'r') as f:
        resources = json.load(f)

    # tokenization of validation dataset
    tokenizer = 'P1_tokenizer' # tokenizer in resources
    text_token = text_tokenization(tokenizer=tokenizer, column=x_col, sequence_lenght=10)
    text_token.resources = resources
    data_token = text_token.apply(df)['0'][0]

    # encoding the text
    text_encode = text_encoding(column=x_col, chunk_size=100, device='cpu')
    data_encode = text_encode.apply(df)
    data_encode_1 = np.concatenate(data_encode['0_model_1_encode'], axis=0)
    data_encode_2 = np.concatenate(data_encode['0_model_3_encode'], axis=0)

    # loading the trained deep learning model
    dl_cls = DeepLearning(
                      mode='testing',
                      model_path=r"C:\Users\em2945\Documents\CWD\test_gemlib\dl_classification\T1_dl_ff_lstm_cnn_model")
    dl_cls.resources = resources
    dl_cls.get_model()
    dl_cls.x_test = [data_token, data_token, data_encode_1, data_encode_2]
    preds = dl_cls.run_testing()