#!/usr/bin/env python
# coding: utf-8

# ## Library

# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import *
import tensorflow as tf
import numpy as np
import pandas as pd
from gensim import models
from konlpy.tag import Mecab
import re


# ## word2vec model (300 dimension)

# In[3]:


mecab_model = models.word2vec.Word2Vec.load('/Users/jaeyoungshin/Desktop/NLP/cnn_2조/mecab model files/word2vec modeling/NaverMovie_mecab300.model')


# ## train/test set

# In[49]:


train_data = pd.read_csv('/Users/jaeyoungshin/Desktop/NLP/cnn_2조/prep for modeling/ratings_train.txt',sep='\t').dropna().reset_index(drop=True)
test_data = pd.read_csv('/Users/jaeyoungshin/Desktop/NLP/cnn_2조/prep for modeling/ratings_test.txt',sep='\t').dropna().reset_index(drop=True)

training_sentences = []
testing_sentences = []


# In[50]:


stopwords_df = pd.read_excel('/Users/jaeyoungshin/Desktop/NLP/cnn_2조/prep for modeling/stopwords.xlsx')
stopwords_list = stopwords_df['불용어'].tolist()


# In[51]:


mecab = Mecab()


# In[52]:


train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-|가-힣]",'')
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-|가-힣]",'')


# In[54]:


documents = []
for item in train_data.document:
    documents.append([word for word in [tokens[0] for tokens in mecab.pos(item)] if word not in stopwords_list])

labels = []
for item in train_data.label:
    labels.append(item)


# In[57]:


test_documents = []
for item in test_data.document:
    test_documents.append([word for word in [tokens[0] for tokens in mecab.pos(item)] if word not in stopwords_list])

test_labels = []
for item in test_data.label:
    test_labels.append(item)


# In[58]:


training_labels_final = np.array(labels)
testing_labels_final = np.array(test_labels)


# ## Padding

# In[59]:


vocab_size = 20000
embedding_dim = 300
max_length = 41
trunc_type='post'
oov_tok = '<OOV>'

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(documents)


# In[60]:


word_index = tokenizer.word_index
text_sequences = tokenizer.texts_to_sequences(documents)
padded = pad_sequences(text_sequences,maxlen=max_length,truncating = 'pre')


# In[61]:


testing_sequences = tokenizer.texts_to_sequences(test_documents)
test_padded = pad_sequences(testing_sequences, maxlen=max_length)


# In[62]:


len(word_index)


# ## Embedding Matrix

# In[63]:


embedding_matrix = np.zeros((vocab_size, embedding_dim))

index = 0 
for word, idx in tokenizer.word_index.items():
    try:
        embedding_vector = mecab_model.wv.__getitem__(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
    except Exception as e:
        pass
        
embedding_matrix.shape


# ## Layers

# In[64]:


sequence_input = tf.keras.Input(shape=(max_length,), dtype='int32')
embedding_layer = tf.keras.layers.Embedding(vocab_size,
                            embedding_dim,
                            input_length=max_length,
                            trainable=False)
embedded_sequences = embedding_layer(sequence_input)
convs = []
filter_sizes = [3,4,5]
for fsz in filter_sizes:
    x = tf.keras.layers.Conv1D(128, fsz, activation='relu',padding='same')(embedded_sequences)
    x = tf.keras.layers.MaxPooling1D()(x)
    convs.append(x)
x = tf.keras.layers.Concatenate(axis=-1)(convs)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(sequence_input, output)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# ## Training

# In[65]:


from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy',min_delta=0.002,patience=1,mode='max')


# In[66]:


num_epochs = 30
history = model.fit(padded, training_labels_final, epochs = num_epochs, validation_data = (test_padded, testing_labels_final), callbacks=[es])
history


# In[67]:


sentence = ["수면제인가 꿀잠잤음","바보같은 영화","조금 기대 이하이긴 했는데 그래도 만족함", "완전 재밌었다", "개꿀잼"]
sequence_exp = tokenizer.texts_to_sequences(sentence)
padded_exp = pad_sequences(sequence_exp, maxlen = max_length, truncating= 'post')
print(model.predict(padded_exp).round(2))


# ## Save model

# In[68]:


model.save('word2vec_mecab300_updated/0.8220_model')


# **load saved model / re-training**

# In[72]:


mecab_model = tf.keras.models.load_model('word2vec_mecab300_updated/0.8220_model')


# In[73]:


from keras.callbacks import EarlyStopping
re_es = EarlyStopping(monitor='val_accuracy',min_delta=0.002,patience=1,mode='max')


# In[74]:


re_history = mecab_model.fit(padded, training_labels_final, epochs = num_epochs, validation_data = (test_padded, testing_labels_final), callbacks=[re_es])
re_history


# **re-training model save (val_accuracy : 0.8437)**

# In[75]:


mecab_model.save('word2vec_mecab300_updated/0.8437_model')


# In[76]:


mecab_remodel = tf.keras.models.load_model('word2vec_mecab300_updated/0.8437_model')


# In[77]:


sentence = ["수면제인가 꿀잠잤음 개꿀","바보같은 영화","바보같은 영화네","조금 기대 이하이긴 했는데 그래도 만족함", "완전 재밌었다", "개꿀잼"]
sequence_exp = tokenizer.texts_to_sequences(sentence)
padded_exp = pad_sequences(sequence_exp, maxlen = max_length, truncating= 'post')
print(mecab_remodel.predict(padded_exp).round(2))


# In[60]:


# 0.7993
sentence = ["완전 추천함","아 겁나 지루했음 개비추..","에라이 때려쳐 이게 영화라고?",'내 인생 영화']
sequence_exp = tokenizer.texts_to_sequences(sentence)
padded_exp = pad_sequences(sequence_exp, maxlen = max_length, truncating= 'post')
print(mecab_remodel.predict(padded_exp).round(2))


# In[78]:


# 0.8437
sentence = ["완전 추천함","아 겁나 지루했음 개비추..","에라이 때려쳐 이게 영화라고?",'내 인생 영화']
sequence_exp = tokenizer.texts_to_sequences(sentence)
padded_exp = pad_sequences(sequence_exp, maxlen = max_length, truncating= 'post')
print(mecab_remodel.predict(padded_exp).round(2))


# In[ ]:




