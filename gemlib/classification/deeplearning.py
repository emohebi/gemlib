from re import L
from gemlib.abstarct.basefunctionality import BaseClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, Input, Layer, Reshape, Conv2D, MaxPool2D, Concatenate, Activation, Add, LayerNormalization
from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, Dropout, GlobalAveragePooling1D, BatchNormalization, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.callbacks import LearningRateScheduler
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import concatenate
import tensorflow.keras.backend as K
from gemlib.validation import utilities
import os
from sklearn.utils import class_weight
import numpy as np
import pandas as pd
from pathlib import Path

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.3):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer 
    (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = self.dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = self.dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def dot_product(self, x, kernel):
        """
        Wrapper for dot product operation, in order to be compatible with both
        Theano and Tensorflow.

        Arguments:
            x: input.
            kernel: weights.

        Returns:
            Tensor
        """
        if K.backend() == 'tensorflow':
            return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
        else:
            return K.dot(x, kernel)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
                'W_regularizer': self.W_regularizer,
                'u_regularizer': self.u_regularizer,
                'b_regularizer': self.b_regularizer,
                'W_constraint': self.W_constraint,
                'u_constraint': self.u_constraint,
                'b_constraint': self.b_constraint,
                'bias': self.bias,
        })
        return config

class DeepLearning(BaseClassifier):

    def init_number_of_classes(self):
        if self.num_classes:
            return
        y_tr = utilities.resolve_caching_load(self.resources, self.y_train)
        y_te = utilities.resolve_caching_load(self.resources, self.y_test)
        self.num_classes = np.unique(np.concatenate([y_tr, y_te], axis=0)).shape[0]
        utilities._info(f'number of classes: {self.num_classes}')

    def get_embedding_layer(self, sequence_lenght, embedding_matrix):
        embedding_layer_ = Embedding(
                                    embedding_matrix.shape[0], # num words
                                    embedding_matrix.shape[1], # embedding dim
                                    weights=[embedding_matrix],
                                    input_length=sequence_lenght,
                                    trainable=False
                                    )
        return embedding_layer_

    def get_text_cnn_layers(self, sequence_lenght, embedding_matrix):
        num_filters = 16
        filter_sizes = (3, 1, 2)
        input_ = Input(shape=(sequence_lenght,))
        embedding_ = self.get_embedding_layer(sequence_lenght, embedding_matrix)(input_)
        expend_shape_ = [embedding_.get_shape().as_list()[1], embedding_.get_shape().as_list()[2], 1]
        embedding_chars_ = Reshape(expend_shape_)(embedding_)
            # conv->max pool
        pooled_outputs_2 = []
        for i, filter_size in enumerate(filter_sizes):
            conv = Conv2D(filters=num_filters, 
                        kernel_size=[filter_size, embedding_matrix.shape[1]],
                        strides=1,
                        padding='valid',
                        activation='relu',
                        kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                        bias_initializer=initializers.constant(value=0.1),
                        name=('conv_2_%d' % i))(embedding_chars_)
            # print("conv-%d: " % i, conv)
            max_pool = MaxPool2D(pool_size=[sequence_lenght - filter_size + 1, 1],
                                strides=(1, 1),
                                padding='valid',
                                name=('max_pool_2_%d' % i))(conv)
            pooled_outputs_2.append(max_pool)
        # print("max_pool-%d: " % i, max_pool)

        # combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool_ = Concatenate(axis=3)(pooled_outputs_2)
        h_pool_flat_ = Reshape([num_filters_total])(h_pool_)

        return input_, h_pool_flat_

    def get_feed_forward_layer(self, input_data, i=0):
        """
        AI is creating summary for get_feed_forward_layer

        Args:
            input_data (numpy.array): input data
            i (int): index of layer. Defaults to 0.

        Returns:
            tensor: a keras layer
        """
        input_ = Input(shape=(input_data.shape[1],), name=f'FF_input_{i}')
        x_1 = Dense(2048, activation='relu', name=f'Dense_1_{i}')(input_)
        x_1 = BatchNormalization(name=f'BatchNorm_1_{i}')(x_1)
        x_1 = Activation('relu', name=f'Act_1_{i}')(x_1)
        # x_0 = Dropout(0.5)(x_0)
        o_1 = Dense(1024, activation='relu', name=f'Dense_2_{i}')(x_1)
        o_1 = BatchNormalization(name=f'BatchNorm_2_{i}')(o_1)
        o_1 = Activation('relu', name=f'Act_2_{i}')(o_1)
        return input_, o_1

    def get_lstm_layers_with_attention(self, sequence_lenght, embedding_matrix):
        lstm_num_layer = 128
        input_lstm = Input(shape=(sequence_lenght,))
        x = self.get_embedding_layer(sequence_lenght, embedding_matrix)(input_lstm)
        x_lstm_1 = Bidirectional(LSTM(int(lstm_num_layer), return_sequences=True))(x)
        x_lstm_2 = Bidirectional(LSTM(int(lstm_num_layer*2), return_sequences=True))(x_lstm_1)
        x_att_lstm_1 = AttentionWithContext()(x_lstm_1)
        x_att_lstm_2 = AttentionWithContext()(x_lstm_2)
        x_avg = GlobalAveragePooling1D()(x_lstm_2)
        x_max = GlobalMaxPooling1D()(x_lstm_2)
        x_con = concatenate([x_avg, x_max, x_att_lstm_2, x_att_lstm_1])
        return input_lstm, x_con

    def get_transformer_layers(self, sequence_lenght, embedding_matrix):
        # embed_dim = 32  # Embedding size for each token
        num_heads = 2  # Number of attention heads
        ff_dim = 1024  # Hidden layer size in feed forward network inside transformer
        input_ = Input(shape=(sequence_lenght,))
        embedding_layer = TokenAndPositionEmbedding(sequence_lenght, 
                                                    embedding_matrix.shape[0],
                                                    embedding_matrix.shape[1])
        x = embedding_layer(input_)
        transformer_block = TransformerBlock(embedding_matrix.shape[1], num_heads, ff_dim)
        x = transformer_block(x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.3)(x)
        x = Dense(1024, activation="relu")(x)
        return input_, x

    def get_dl_fff_lstm_cnn_trans_model(self, ff_data_1, ff_data_2, ff_data_3):
        lstm_seq_len, cnn_seq_len, trans_seq_len = self.sequence_lenght
        lstm_embedding_matrix = utilities.resolve_caching_load(self.resources, self.embedding_matrix[0])
        cnn_embedding_matrix = utilities.resolve_caching_load(self.resources, self.embedding_matrix[1])
        trans_embedding_matrix = utilities.resolve_caching_load(self.resources, self.embedding_matrix[2])
        cnn_in, cnn_out = self.get_text_cnn_layers(cnn_seq_len, cnn_embedding_matrix)
        lstm_in, lstm_out = self.get_lstm_layers_with_attention(lstm_seq_len, lstm_embedding_matrix)
        ff_in_1, ff_out_1 = self.get_feed_forward_layer(ff_data_1, 0)
        ff_in_2, ff_out_2 = self.get_feed_forward_layer(ff_data_2, 1)
        ff_in_3, ff_out_3 = self.get_feed_forward_layer(ff_data_3, 2)
        tb_in, tb_out = self.get_transformer_layers(trans_seq_len, trans_embedding_matrix)

        d_ = Concatenate()([cnn_out, ff_out_3, ff_out_2, ff_out_1, lstm_out, tb_out])
        d_ = Dense(1024, activation='relu')(d_)
        d_ = BatchNormalization()(d_)
        # add dropout
        dropout = Dropout(0.3)(d_)

        # output layer
        output = Dense(self.num_classes, activation='softmax')(dropout)
        model = Model(inputs=[
                            lstm_in,
                            cnn_in,
                            tb_in,
                            ff_in_1, 
                            ff_in_2,
                            ff_in_3
                            ], outputs=output, name=self.model_name)
        model.compile(
                loss='categorical_crossentropy',
                optimizer=Adam(lr=self.lr),
                metrics=['accuracy']
        )
        return model

    def get_dl_fff_cnn_trans_model(self, ff_data_1, ff_data_2, ff_data_3):
        cnn_seq_len, trans_seq_len = self.sequence_lenght
        cnn_embedding_matrix = utilities.resolve_caching_load(self.resources, self.embedding_matrix[0])
        trans_embedding_matrix = utilities.resolve_caching_load(self.resources, self.embedding_matrix[1])
        cnn_in, cnn_out = self.get_text_cnn_layers(cnn_seq_len, cnn_embedding_matrix)
        ff_in_1, ff_out_1 = self.get_feed_forward_layer(ff_data_1, 0)
        ff_in_2, ff_out_2 = self.get_feed_forward_layer(ff_data_2, 1)
        ff_in_3, ff_out_3 = self.get_feed_forward_layer(ff_data_3, 2)
        tb_in, tb_out = self.get_transformer_layers(trans_seq_len, trans_embedding_matrix)

        d_ = Concatenate()([cnn_out, ff_out_3, ff_out_2, ff_out_1, tb_out])
        d_ = Dense(1024, activation='relu')(d_)
        d_ = BatchNormalization()(d_)
        # add dropout
        dropout = Dropout(0.3)(d_)

        # output layer
        output = Dense(self.num_classes, activation='softmax')(dropout)
        model = Model(inputs=[
                            cnn_in,
                            tb_in,
                            ff_in_1, 
                            ff_in_2,
                            ff_in_3
                            ], outputs=output, name=self.model_name)
        model.compile(
                loss='categorical_crossentropy',
                optimizer=Adam(lr=self.lr),
                metrics=['accuracy']
        )
        return model

    def get_dl_fff_lstm_cnn_model(self, ff_data_1, ff_data_2, ff_data_3):
        lstm_seq_len, cnn_seq_len = self.sequence_lenght
        lstm_embedding_matrix = utilities.resolve_caching_load(self.resources, self.embedding_matrix[0])
        cnn_embedding_matrix = utilities.resolve_caching_load(self.resources, self.embedding_matrix[1])
        cnn_in, cnn_out = self.get_text_cnn_layers(cnn_seq_len, cnn_embedding_matrix)
        lstm_in, lstm_out = self.get_lstm_layers_with_attention(lstm_seq_len, lstm_embedding_matrix)
        ff_in_1, ff_out_1 = self.get_feed_forward_layer(ff_data_1, 0)
        ff_in_2, ff_out_2 = self.get_feed_forward_layer(ff_data_2, 1)
        ff_in_3, ff_out_3 = self.get_feed_forward_layer(ff_data_3, 2)

        d_ = Concatenate()([cnn_out, ff_out_3, ff_out_2, ff_out_1, lstm_out])
        d_ = Dense(1024, activation='relu')(d_)
        d_ = BatchNormalization()(d_)
        # add dropout
        dropout = Dropout(0.3)(d_)

        # output layer
        output = Dense(self.num_classes, activation='softmax')(dropout)
        model = Model(inputs=[
                            lstm_in,
                            cnn_in,  
                            ff_in_1, 
                            ff_in_2,
                            ff_in_3
                            ], outputs=output, name=self.model_name)
        model.compile(
                loss='categorical_crossentropy',
                optimizer=Adam(lr=self.lr),
                metrics=['accuracy']
        )
        return model

    def get_dl_ff_lstm_cnn_model(self, ff_data_1, ff_data_2):
        lstm_seq_len, cnn_seq_len = self.sequence_lenght
        lstm_embedding_matrix = utilities.resolve_caching_load(self.resources, self.embedding_matrix[0])
        cnn_embedding_matrix = utilities.resolve_caching_load(self.resources, self.embedding_matrix[1])
        cnn_in, cnn_out = self.get_text_cnn_layers(cnn_seq_len, cnn_embedding_matrix)
        lstm_in, lstm_out = self.get_lstm_layers_with_attention(lstm_seq_len, lstm_embedding_matrix)
        ff_in_1, ff_out_1 = self.get_feed_forward_layer(ff_data_1, 0)
        ff_in_2, ff_out_2 = self.get_feed_forward_layer(ff_data_2, 1)

        d_ = Concatenate()([cnn_out, ff_out_2, ff_out_1, lstm_out])
        d_ = Dense(1024, activation='relu')(d_)
        d_ = BatchNormalization()(d_)
        # add dropout
        dropout = Dropout(0.3)(d_)

        # output layer
        output = Dense(self.num_classes, activation='softmax')(dropout)
        model = Model(inputs=[
                            lstm_in,
                            cnn_in,
                            ff_in_1, 
                            ff_in_2
                            ], outputs=output, name=self.model_name)
        model.compile(
                loss='categorical_crossentropy',
                optimizer=Adam(lr=self.lr),
                metrics=['accuracy']
        )
        return model

    def get_dl_ff_cnn_model(self, ff_data_1, ff_data_2):
        cnn_seq_len = self.sequence_lenght[0]
        cnn_embedding_matrix = utilities.resolve_caching_load(self.resources, self.embedding_matrix[0])
        cnn_in, cnn_out = self.get_text_cnn_layers(cnn_seq_len, cnn_embedding_matrix)
        ff_in_1, ff_out_1 = self.get_feed_forward_layer(ff_data_1, 0)
        ff_in_2, ff_out_2 = self.get_feed_forward_layer(ff_data_2, 1)

        d_ = Concatenate()([cnn_out, ff_out_2, ff_out_1])
        d_ = Dense(1024, activation='relu')(d_)
        d_ = BatchNormalization()(d_)
        # add dropout
        dropout = Dropout(0.3)(d_)

        # output layer
        output = Dense(self.num_classes, activation='softmax')(dropout)
        model = Model(inputs=[
                            cnn_in,
                            ff_in_1, 
                            ff_in_2
                            ], outputs=output, name=self.model_name)
        model.compile(
                loss='categorical_crossentropy',
                optimizer=Adam(lr=self.lr),
                metrics=['accuracy']
        )
        return model

    def get_model(self):
        if self.model_name == 'dl_fff_lstm_cnn_trans':  
            _, _, _, ff_data_1, ff_data_2, ff_data_3 = utilities.resolve_caching_load(self.resources, self.x_train)
            self.init_number_of_classes()
            self.model = self.get_dl_fff_lstm_cnn_trans_model(ff_data_1, ff_data_2, ff_data_3)
            utilities._info(f'{self.model.summary()}')
        elif os.path.isdir(self.model_path) and self.model_path.find('dl_fff_lstm_cnn_trans') != -1:
            self.model = load_model(self.model_path, custom_objects={'AttentionWithContext':AttentionWithContext})
            utilities._info(f'model loaded from {self.model_path} with model_name: {self.model.name}')
            self.model_name = 'dl_fff_lstm_cnn_trans'
            utilities._info(f'{self.model.summary()}')
        elif self.model_name == 'dl_fff_cnn_trans':  
            _, _, ff_data_1, ff_data_2, ff_data_3 = utilities.resolve_caching_load(self.resources, self.x_train)
            self.init_number_of_classes()
            self.model = self.get_dl_fff_cnn_trans_model(ff_data_1, ff_data_2, ff_data_3)
            utilities._info(f'{self.model.summary()}')
        elif os.path.isdir(self.model_path) and self.model_path.find('dl_fff_cnn_trans') != -1:
            self.model = load_model(self.model_path)
            utilities._info(f'model loaded from {self.model_path} with model_name: {self.model.name}')
            self.model_name = 'dl_fff_cnn_trans'
            utilities._info(f'{self.model.summary()}')
        elif self.model_name == 'dl_fff_lstm_cnn':  
            _, _, ff_data_1, ff_data_2, ff_data_3 = utilities.resolve_caching_load(self.resources, self.x_train)
            self.init_number_of_classes()
            self.model = self.get_dl_fff_lstm_cnn_model(ff_data_1, ff_data_2, ff_data_3)
            utilities._info(f'{self.model.summary()}')
        elif os.path.isdir(self.model_path) and self.model_path.find('dl_fff_lstm_cnn') != -1:
            self.model = load_model(self.model_path, custom_objects={'AttentionWithContext':AttentionWithContext})
            utilities._info(f'model loaded from {self.model_path} with model_name: {self.model.name}')
            self.model_name = 'dl_fff_lstm_cnn'
            utilities._info(f'{self.model.summary()}')
        elif self.model_name == 'dl_ff_lstm_cnn':  
            _, _, ff_data_1, ff_data_2 = utilities.resolve_caching_load(self.resources, self.x_train)
            self.init_number_of_classes()
            self.model = self.get_dl_ff_lstm_cnn_model(ff_data_1, ff_data_2)
            utilities._info(f'{self.model.summary()}')
        elif os.path.isdir(self.model_path) and self.model_path.find('dl_ff_lstm_cnn') != -1:
            self.model = load_model(self.model_path, custom_objects={'AttentionWithContext':AttentionWithContext})
            utilities._info(f'model loaded from {self.model_path} with model_name: {self.model.name}')
            self.model_name = 'dl_ff_lstm_cnn'
            utilities._info(f'{self.model.summary()}')
        elif self.model_name == 'dl_ff_cnn':  
            _, ff_data_1, ff_data_2 = utilities.resolve_caching_load(self.resources, self.x_train)
            self.init_number_of_classes()
            self.model = self.get_dl_ff_cnn_model(ff_data_1, ff_data_2)
            utilities._info(f'{self.model.summary()}')
        elif os.path.isdir(self.model_path) and self.model_path.find('dl_ff_cnn') != -1:
            self.model = load_model(self.model_path)
            utilities._info(f'model loaded from {self.model_path} with model_name: {self.model.name}')
            self.model_name = 'dl_ff_cnn'
            utilities._info(f'{self.model.summary()}')
        else:
            utilities._error(NotImplementedError, f'model:{self.model_name} can not be loaded or is not implemented yet!!!')

    def run(self):
        self.get_model()
        if self.mode == 'training':
            model_path = str(Path(self.dirpath) / f'{self.name}_{self.model_name}_model')
            self.run_training()
            # save_model(self.model, model_path)
            # utilities._info(f'model saved: {model_path}')
            return model_path
        elif self.mode == 'testing':
            return self.run_testing()
        return

    def decay_schedule(self, epoch, lr):
        # decay by 0.1 every 5 epochs; use `% 1` to decay after each epoch
        if (epoch % self.decay_n_epoch == 0) and (epoch != 0):
            lr = lr * 0.1
        return lr
    
    def run_training(self):
        if self.model_name == 'dl_fff_lstm_cnn_trans':
            x_1, x_2, x_3, ff_data_1, ff_data_2, ff_data_3 = utilities.resolve_caching_load(self.resources, self.x_train)
            y_ = utilities.resolve_caching_load(self.resources, self.y_train)
            
            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_), y_)
            cl_weights = {i : class_weights[i] for i in range(class_weights.shape[0])}

            x_test_1, x_test_2, x_test_3, ff_test_data_1, ff_test_data_2, ff_test_data_3 = utilities.resolve_caching_load(self.resources, self.x_test)
            y_test = utilities.resolve_caching_load(self.resources, self.y_test)

            model_path = str(Path(self.dirpath) / f'{self.name}_{self.model_name}_model')
            mcp_save = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
            es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
            lr_scheduler = LearningRateScheduler(self.decay_schedule)
            return self.model.fit(
                        [
                            np.vstack(1*[x_1]), 
                            np.vstack(1*[x_2]),
                            np.vstack(1*[x_3]),
                            np.vstack(1*[ff_data_1]),  
                            np.vstack(1*[ff_data_2]),
                            np.vstack(1*[ff_data_3])
                        ],
                        to_categorical(np.hstack(1*[y_]), self.num_classes),
                        batch_size=self.batch_size,
                        epochs=self.epoch,
                        validation_data = ([
                                            x_test_1, 
                                            x_test_2,
                                            x_test_3,
                                            ff_test_data_1,
                                            ff_test_data_2,
                                            ff_test_data_3
                                            ], to_categorical(y_test, self.num_classes)),
                        callbacks=[es, mcp_save, lr_scheduler],
                        class_weight=cl_weights
                        )
        elif self.model_name == 'dl_fff_cnn_trans':
            x_1, x_2, ff_data_1, ff_data_2, ff_data_3 = utilities.resolve_caching_load(self.resources, self.x_train)
            y_ = utilities.resolve_caching_load(self.resources, self.y_train)
            
            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_), y_)
            cl_weights = {i : class_weights[i] for i in range(class_weights.shape[0])}

            x_test_1, x_test_2, ff_test_data_1, ff_test_data_2, ff_test_data_3 = utilities.resolve_caching_load(self.resources, self.x_test)
            y_test = utilities.resolve_caching_load(self.resources, self.y_test)

            model_path = str(Path(self.dirpath) / f'{self.name}_{self.model_name}_model')
            mcp_save = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
            es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
            return self.model.fit(
                        [
                            np.vstack(1*[x_1]), 
                            np.vstack(1*[x_2]),
                            np.vstack(1*[ff_data_1]),  
                            np.vstack(1*[ff_data_2]),
                            np.vstack(1*[ff_data_3])
                        ],
                        to_categorical(np.hstack(1*[y_]), self.num_classes),
                        batch_size=self.batch_size,
                        epochs=self.epoch,
                        validation_data = ([
                                            x_test_1, 
                                            x_test_2,
                                            ff_test_data_1,
                                            ff_test_data_2,
                                            ff_test_data_3
                                            ], to_categorical(y_test, self.num_classes)),
                        callbacks=[es, mcp_save],
                        class_weight=cl_weights
                        )
        elif self.model_name == 'dl_fff_lstm_cnn':
            x_1, x_2, ff_data_1, ff_data_2, ff_data_3 = utilities.resolve_caching_load(self.resources, self.x_train)
            y_ = utilities.resolve_caching_load(self.resources, self.y_train)
            
            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_), y_)
            cl_weights = {i : class_weights[i] for i in range(class_weights.shape[0])}

            x_test_1, x_test_2, ff_test_data_1, ff_test_data_2, ff_test_data_3 = utilities.resolve_caching_load(self.resources, self.x_test)
            y_test = utilities.resolve_caching_load(self.resources, self.y_test)

            model_path = str(Path(self.dirpath) / f'{self.name}_{self.model_name}_model')
            mcp_save = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
            es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
            return self.model.fit(
                        [
                            np.vstack(1*[x_1]), 
                            np.vstack(1*[x_2]),
                            np.vstack(1*[ff_data_1]),  
                            np.vstack(1*[ff_data_2]),
                            np.vstack(1*[ff_data_3])
                        ],
                        to_categorical(np.hstack(1*[y_]), self.num_classes),
                        batch_size=self.batch_size,
                        epochs=self.epoch,
                        validation_data = ([
                                            x_test_1, 
                                            x_test_2,
                                            ff_test_data_1,
                                            ff_test_data_2,
                                            ff_test_data_3
                                            ], to_categorical(y_test, self.num_classes)),
                        callbacks=[es, mcp_save],
                        class_weight=cl_weights
                        )
        elif self.model_name == 'dl_ff_lstm_cnn':
            x_1, x_2, ff_data_1, ff_data_2 = utilities.resolve_caching_load(self.resources, self.x_train)
            y_ = utilities.resolve_caching_load(self.resources, self.y_train)
            
            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_), y_)
            cl_weights = {i : class_weights[i] for i in range(class_weights.shape[0])}

            x_test_1, x_test_2, ff_test_data_1, ff_test_data_2 = utilities.resolve_caching_load(self.resources, self.x_test)
            y_test = utilities.resolve_caching_load(self.resources, self.y_test)

            model_path = str(Path(self.dirpath) / f'{self.name}_{self.model_name}_model')
            mcp_save = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
            es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
            return self.model.fit(
                        [
                            np.vstack(1*[x_1]), 
                            np.vstack(1*[x_2]),
                            np.vstack(1*[ff_data_1]),  
                            np.vstack(1*[ff_data_2])
                        ],
                        to_categorical(np.hstack(1*[y_]), self.num_classes),
                        batch_size=self.batch_size,
                        epochs=self.epoch,
                        validation_data = ([
                                            x_test_1, 
                                            x_test_2,
                                            ff_test_data_1,
                                            ff_test_data_2
                                            ], to_categorical(y_test, self.num_classes)),
                        callbacks=[es, mcp_save],
                        class_weight=cl_weights
                        )
        elif self.model_name == 'dl_ff_cnn':
            x_1, ff_data_1, ff_data_2 = utilities.resolve_caching_load(self.resources, self.x_train)
            y_ = utilities.resolve_caching_load(self.resources, self.y_train)
            
            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_), y_)
            cl_weights = {i : class_weights[i] for i in range(class_weights.shape[0])}

            x_test_1, ff_test_data_1, ff_test_data_2 = utilities.resolve_caching_load(self.resources, self.x_test)
            y_test = utilities.resolve_caching_load(self.resources, self.y_test)

            model_path = str(Path(self.dirpath) / f'{self.name}_{self.model_name}_model')
            mcp_save = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
            es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
            return self.model.fit(
                        [
                            np.vstack(1*[x_1]),
                            np.vstack(1*[ff_data_1]),  
                            np.vstack(1*[ff_data_2])
                        ],
                        to_categorical(np.hstack(1*[y_]), self.num_classes),
                        batch_size=self.batch_size,
                        epochs=self.epoch,
                        validation_data = ([
                                            x_test_1, 
                                            ff_test_data_1,
                                            ff_test_data_2
                                            ], to_categorical(y_test, self.num_classes)),
                        callbacks=[es, mcp_save],
                        class_weight=cl_weights
                        )
        else:
            utilities._error(NotImplementedError, f'model:{self.model_name} can not be loaded or is not implemented yet!!!')

    def run_testing(self):
        x_test = utilities.resolve_caching_load(self.resources, self.x_test)
        preds = self.model.predict(x_test)
        df = pd.DataFrame(preds.argsort()[:,-self.top_n:]).T.unstack().to_frame().reset_index().rename({'level_0':'index', 
                                                                                                    'level_1':'rank',
                                                                                                    0:'prediction'}, axis='columns')
        df_p = pd.DataFrame(np.sort(preds)[:,-self.top_n:]).T.unstack().to_frame().reset_index().rename({'level_0':'index', 
                                                                                                    'level_1':'rank',
                                                                                                    0:'probability'}, axis='columns')
        df = df.merge(df_p, on=['index', 'rank'])
        _mapping = utilities.resolve_caching_load(self.resources, self.y_col_map)
        if not isinstance(_mapping, dict) and _mapping is not None:
            _mapping = {c:_mapping[c] for c in range(_mapping.shape[0])}
            df['prediction'] = df['prediction'].map(_mapping)
        return df