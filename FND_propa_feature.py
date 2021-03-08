#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import *
from official import nlp
import official.nlp.optimization
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import re


# In[2]:


def load_propa_data():
    dfUDN = pd.read_csv('originalDataset/propa/UDN-bootstrap-checked.tsv', sep='\t', names=['宣傳手法', '標記詞', '句子'])
    dfLT = pd.read_csv('originalDataset/propa/LT-bootstrap-checked.tsv', sep='\t', names=['宣傳手法', '標記詞', '句子'])
    dfUDN_new = pd.read_csv('originalDataset/propa/UDN-bootstrap-checked-20210209.tsv', sep='\t', names=['宣傳手法', '標記詞', '句子'])
    dfLT_new = pd.read_csv('originalDataset/propa/LT-bootstrap-checked-20210209.tsv', sep='\t', names=['宣傳手法', '標記詞', '句子'])
    dfPropa = pd.concat([dfUDN, dfLT, dfUDN_new, dfLT_new], ignore_index=True)
    #dfPropa = pd.concat([dfUDN, dfLT], ignore_index=True)
    return dfPropa

def encode_data(contexts, tokenizer, sent_len):
    input_ids, attention_mask = [], []
    for i in range(len(contexts)):
        inputs = tokenizer.encode_plus(contexts[i],add_special_tokens=True, max_length=sent_len, pad_to_max_length=True,
                    return_attention_mask=True, return_token_type_ids=False, truncation=True)
        input_ids.append(inputs['input_ids'])
        attention_mask.append(inputs['attention_mask'])
    
    return np.asarray(input_ids, dtype='int32'), np.asarray(attention_mask, dtype='int32')

def remove_not_Ch_Eng(cont):
    # chinese unicode range: [0x4E00,0x9FA5]
    rule = u'[\u4E00-\u9FA5\w]'
    for i in range(len(cont)):
        pChEng = re.compile(rule).findall(cont[i])
        ChEngText = "".join(pChEng)
        cont[i] = ChEngText
    return cont

def plot_learning_curve(hist):
    pd.DataFrame(hist.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
    return


# In[3]:


def build_model_propa(base, lr, epochs, batchSize, train_data_size, sent_len):
    model = TFBertModel.from_pretrained(base, return_dict=True)
    inp1 = tf.keras.Input(shape=(sent_len,), dtype=tf.int32, name='input_ids')
    inp2 = tf.keras.Input(shape=(sent_len,), dtype=tf.int32, name='attention_mask')
    outDict = model([inp1, inp2])
    #print(outDict.keys())
    # >>> odict_keys(['last_hidden_state', 'pooler_output'])
    pooler_out = outDict['pooler_output']
    hid1 = tf.keras.layers.Dense(64, activation='relu')(pooler_out)
    result = tf.keras.layers.Dense(1, activation='sigmoid')(hid1)
    #spe = steps_per_epoch
    spe = int(train_data_size/batchSize)
    #nts = num_train_steps
    nts = spe * epochs
    #ws = warmup_steps
    ws = int(epochs * train_data_size * 0.1 / batchSize)
    opWarm = nlp.optimization.create_optimizer(lr, num_train_steps=nts,
                                                  num_warmup_steps=ws)
    model = tf.keras.Model(inputs=[inp1, inp2], outputs=result)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    metric = tf.keras.metrics.BinaryAccuracy('accuracy')
    model.compile(optimizer=opWarm, loss=loss_fn, metrics=[metric])
    
    model.summary()
    
    return model


# In[4]:


def build_model_baseline_2way(base, lr, epochs, batch_size, train_data_size):
    model = TFBertModel.from_pretrained(base, return_dict=True)
    inp1 = tf.keras.Input(shape=(160,), dtype=tf.int32, name='input_ids')
    inp2 = tf.keras.Input(shape=(160,), dtype=tf.int32, name='attention_mask')
    outDict = model([inp1, inp2])
    #print(outDict.keys())
    # >>> odict_keys(['last_hidden_state', 'pooler_output'])
    pooler_out = outDict['pooler_output']
    hid1 = tf.keras.layers.Dense(64, activation='relu')(pooler_out)
    result = tf.keras.layers.Dense(2, activation='softmax')(hid1)
    #spe = steps_per_epoch
    spe = int(train_data_size/batch_size)
    #nts = num_train_steps
    nts = spe * epochs
    #ws = warmup_steps
    ws = int(epochs * train_data_size * 0.1 / batch_size)
    opWarm = nlp.optimization.create_optimizer(lr, num_train_steps=nts,
                                                  num_warmup_steps=ws)
    model = tf.keras.Model(inputs=[inp1, inp2], outputs=result)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    metric = tf.keras.metrics.BinaryAccuracy('accuracy')
    model.compile(optimizer=opWarm, loss=loss_fn, metrics=[metric])
    
    model.summary()
    
    return model

def build_model_propa_2way(base, lr, epochs, batch_size, train_data_size):
    model = TFBertModel.from_pretrained(base, return_dict=True)
    inp1 = tf.keras.Input(shape=(160,), dtype=tf.int32, name='input_ids')
    inp2 = tf.keras.Input(shape=(160,), dtype=tf.int32, name='attention_mask')
    inp4 = tf.keras.Input(shape=(10,), dtype=tf.float32, name='propa_01')
    outDict = model([inp1, inp2])
    #print(outDict.keys())
    # >>> odict_keys(['last_hidden_state', 'pooler_output'])
    #print(outDict['last_hidden_state'].shape)
    pooler_out = outDict['pooler_output']
    concat_inp = tf.keras.layers.concatenate([pooler_out, inp4])
    hid1 = tf.keras.layers.Dense(64, activation='relu')(concat_inp)
    result = tf.keras.layers.Dense(2, activation='softmax')(hid1)
    #spe = steps_per_epoch
    spe = int(train_data_size/batch_size)
    #nts = num_train_steps
    nts = spe * epochs
    #ws = warmup_steps
    ws = int(epochs * train_data_size * 0.1 / batch_size)
    opWarm = nlp.optimization.create_optimizer(lr, num_train_steps=nts,
                                                  num_warmup_steps=ws)
    model = tf.keras.Model(inputs=[inp1, inp2, inp4], outputs=result)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    metric = tf.keras.metrics.BinaryAccuracy('accuracy')
    model.compile(optimizer=opWarm, loss=loss_fn, metrics=[metric])
    
    model.summary()
    
    return model

def build_model_baseline_3way(base, lr, epochs, batch_size, train_data_size):
    model = TFBertModel.from_pretrained(base, return_dict=True)
    inp1 = tf.keras.Input(shape=(160,), dtype=tf.int32, name='input_ids')
    inp2 = tf.keras.Input(shape=(160,), dtype=tf.int32, name='attention_mask')
    outDict = model([inp1, inp2])
    #print(outDict.keys())
    # >>> odict_keys(['last_hidden_state', 'pooler_output'])
    pooler_out = outDict['pooler_output']
    hid1 = tf.keras.layers.Dense(64, activation='relu')(pooler_out)
    result = tf.keras.layers.Dense(3, activation='softmax')(hid1)
    #spe = steps_per_epoch
    spe = int(train_data_size/batch_size)
    #nts = num_train_steps
    nts = spe * epochs
    #ws = warmup_steps
    ws = int(epochs * train_data_size * 0.1 / batch_size)
    opWarm = nlp.optimization.create_optimizer(lr, num_train_steps=nts,
                                                  num_warmup_steps=ws)
    model = tf.keras.Model(inputs=[inp1, inp2], outputs=result)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    metric = tf.keras.metrics.BinaryAccuracy('accuracy')
    model.compile(optimizer=opWarm, loss=loss_fn, metrics=[metric])
    
    model.summary()
    
    return model

def build_model_propa_3way(base, lr, epochs, batch_size, train_data_size):
    model = TFBertModel.from_pretrained(base, return_dict=True)
    inp1 = tf.keras.Input(shape=(160,), dtype=tf.int32, name='input_ids')
    inp2 = tf.keras.Input(shape=(160,), dtype=tf.int32, name='attention_mask')
    inp4 = tf.keras.Input(shape=(10,), dtype=tf.float32, name='propa_01')
    outDict = model([inp1, inp2])
    #print(outDict.keys())
    # >>> odict_keys(['last_hidden_state', 'pooler_output'])
    pooler_out = outDict['pooler_output']
    concat_inp = tf.keras.layers.concatenate([pooler_out, inp4])
    hid1 = tf.keras.layers.Dense(64, activation='relu')(concat_inp)
    result = tf.keras.layers.Dense(3, activation='softmax')(hid1)
    #spe = steps_per_epoch
    spe = int(train_data_size/batch_size)
    #nts = num_train_steps
    nts = spe * epochs
    #ws = warmup_steps
    ws = int(epochs * train_data_size * 0.1 / batch_size)
    opWarm = nlp.optimization.create_optimizer(lr, num_train_steps=nts,
                                                  num_warmup_steps=ws)
    model = tf.keras.Model(inputs=[inp1, inp2, inp4], outputs=result)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    metric = tf.keras.metrics.BinaryAccuracy('accuracy')
    model.compile(optimizer=opWarm, loss=loss_fn, metrics=[metric])
    
    model.summary()
    
    return model


# In[12]:


def build_model_embedding(base):
    model = TFBertModel.from_pretrained(base, return_dict=True)
    #model.layers[0].trainable = False
    inp1 = tf.keras.Input(shape=(16,), dtype=tf.int32, name='input_ids')
    inp2 = tf.keras.Input(shape=(16,), dtype=tf.int32, name='attention_mask')
    outDict = model([inp1, inp2])
    #print(outDict.keys())
    # >>> odict_keys(['last_hidden_state', 'pooler_output'])
    pooler_out = outDict['pooler_output']
    #print(outDict['pooler_output'].shape)
    # >>> (None, 768)
    model = tf.keras.Model(inputs=[inp1, inp2], outputs=pooler_out)
    
    return model

def build_model_try(sent_num, lr):
    inp1 = tf.keras.Input(shape=(768*sent_num,), dtype=tf.int32, name='sentence1')
    inp2 = tf.keras.Input(shape=(sent_num,), dtype=tf.float32, name='propa_01')
    hid = tf.keras.layers.Dense(768, activation='relu')(inp1)
    hid = tf.keras.layers.Dense(512, activation='relu')(hid)
    hid = tf.keras.layers.Dropout(0.1)(hid)
    hid = tf.keras.layers.Dense(128, activation='relu')(hid)
    hid = tf.keras.layers.Dense(32, activation='relu')(hid)
    concat = tf.keras.layers.concatenate([hid, inp2])
    hid = tf.keras.layers.Dense(8, activation='relu')(hid)
    out = tf.keras.layers.Dense(2, activation='softmax')(hid)
    model = tf.keras.Model(inputs=inp1, outputs=out)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    met = tf.keras.metrics.BinaryAccuracy('accuracy')
    model.compile(optimizer=opt, loss=loss_fn, metrics=met)
    model.summary()
    
    return model


# ### Main function

# In[6]:


# main funciton
model_base = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_base)
sentence_len = 16


# In[7]:


# generate propa training data
dfPropa = load_propa_data()
dfPropa['宣傳手法'] = dfPropa['宣傳手法'].apply(lambda x: 0 if x == 'X' else 1)
neg, pos = np.bincount(dfPropa['宣傳手法'])
propa_lab = dfPropa['宣傳手法'].to_numpy().astype(np.float32)
#propa_lab = tf.keras.utils.to_categorical(propa_lab, num_classes=2)
propa_contexts = dfPropa['句子'].to_numpy()
# remove all not chinese, english and number tokens
filted_propa_contexts = remove_not_Ch_Eng(propa_contexts)

# prepare training & testing data
sh_propa_cont, sh_propa_lab = shuffle(filted_propa_contexts, propa_lab, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(sh_propa_cont, sh_propa_lab, test_size=0.2)
train_ii, train_am = encode_data(X_train, tokenizer, sentence_len)
test_ii, test_am = encode_data(X_test, tokenizer, sentence_len)

# build & train propa classify model
weight_for_0 = (1 / neg)*(neg+pos)/2.0 
weight_for_1 = (1 / pos)*(neg+pos)/2.0
learning_rate = 3e-5
epochs = 4
batch_size = 16
c_weight = {0: weight_for_0, 1: weight_for_1}
train_data_size = len(y_train)
propa_param = [model_base, learning_rate, epochs, batch_size, train_data_size, sentence_len]
model = build_model_propa(model_base, learning_rate, epochs, 
                          batch_size, train_data_size, sentence_len)
history = model.fit([train_ii, train_am],
                   y_train, epochs=epochs, batch_size=batch_size, class_weight=c_weight)
plot_learning_curve(history)

# score the propa classify model
y_pred = model.predict([test_ii, test_am])
y_pred_bool = y_pred.round()
print(classification_report(y_test, y_pred_bool))

# save the model for later use
model.save_weights('saved_model_weight/propa_feature')


# In[8]:


# get 2-way datasets
labels_2w = []
articles = []
max_sent_num = 10
line_num = 0
with open('processed_2way_cofact.txt', 'r') as f:
    sentences = []
    for line in f:
        lab, sent = line.strip().split('|')
        if sent == '<SentencePad>':
            sent = ''
        if line_num%max_sent_num == 0:
            labels_2w.append(float(lab))
        sentences.append(sent)
        line_num+=1
        if line_num%max_sent_num == 0:
            articles.append(sentences)
            sentences = []
input_ids, attention_mask, propa_2w = [], [], []
for a in tqdm(range(len(articles))):
    ii, am = encode_data(articles[a], tokenizer, sentence_len)
    input_ids.append(ii)
    attention_mask.append(am)
    p = model.predict([ii, am])
    propa_2w.append(p)
for i in range(len(propa_2w)):
    propa_2w[i] = np.concatenate(propa_2w[i])
propa_2w = np.array(propa_2w)
iids = []
am = []
for i in range(len(input_ids)):
    iids.append(np.concatenate(input_ids[i]))
for i in range(len(attention_mask)):
    am.append(np.concatenate(attention_mask[i]))
lab_2w = tf.keras.utils.to_categorical(labels_2w, num_classes=2)
iids = np.array(iids)
am = np.array(am)
tr_ii, te_ii, tr_am, te_am, tr_propa, te_propa, tr_lab, te_lab = train_test_split(iids, 
                               am, propa_2w, lab_2w)


# In[8]:


# build & train 2-way baseline model
learning_rate = 3e-5
epochs = 4
batch_size = 16
train_data_size = len(tr_lab)
model = build_model_baseline_2way(model_base, learning_rate, epochs, batch_size, train_data_size)
history = model.fit([tr_ii, tr_am], tr_lab, epochs=epochs, batch_size=batch_size)
plot_learning_curve(history)

# score the baseline model
y_pred = model.predict([te_ii, te_am])
y_pred_bool = y_pred.round()
print(classification_report(te_lab, y_pred_bool))


# In[9]:


# build & train add feature 2-way model
learning_rate = 3e-5
epochs = 4
batch_size = 16
train_data_size = len(tr_lab)
model = build_model_propa_2way(model_base, learning_rate, epochs, batch_size, train_data_size)
history = model.fit([tr_ii, tr_am, tr_propa],
                   tr_lab, epochs=epochs, batch_size=batch_size)
plot_learning_curve(history)

# score the baseline model
y_pred = model.predict([te_ii, te_am, te_propa])
y_pred_bool = y_pred.round()
print(classification_report(te_lab, y_pred_bool))


# In[10]:


# get 3-way datasets
labels_3w = []
articles = []
max_sent_num = 10
line_num = 0
with open('processed_3way_cofact.txt', 'r') as f:
    sentences = []
    for line in f:
        lab, sent = line.strip().split('|')
        if sent == '<SentencePad>':
            sent = ''
        if line_num%max_sent_num == 0:
            labels_3w.append(float(lab))
        sentences.append(sent)
        line_num+=1
        if line_num%max_sent_num == 0:
            articles.append(sentences)
            sentences = []
model = build_model_propa(propa_param[0], propa_param[1], propa_param[2], propa_param[3], 
                         propa_param[4], propa_param[5])
model.load_weights('saved_model_weight/propa_feature')

input_ids, attention_mask, propa_3w = [], [], []
for a in tqdm(range(len(articles))):
    ii, am = encode_data(articles[a], tokenizer, sentence_len)
    input_ids.append(ii)
    attention_mask.append(am)
    p = model.predict([ii, am])
    propa_3w.append(p)
for i in range(len(propa_3w)):
    propa_3w[i] = np.concatenate(propa_3w[i])
propa_3w = np.array(propa_3w)
iids = []
am = []
for i in range(len(input_ids)):
    iids.append(np.concatenate(input_ids[i]))
for i in range(len(attention_mask)):
    am.append(np.concatenate(attention_mask[i]))
lab_3w = tf.keras.utils.to_categorical(labels_3w, num_classes=3)
iids = np.array(iids)
am = np.array(am)
tr_ii, te_ii, tr_am, te_am, tr_propa, te_propa, tr_lab, te_lab = train_test_split(iids, 
                               am, propa_3w, lab_3w)


# In[11]:


# build & train 3-way baseline model
learning_rate = 3e-5
epochs = 4
batch_size = 16
train_data_size = len(tr_lab)
model = build_model_baseline_3way(model_base, learning_rate, epochs, batch_size, train_data_size)
history = model.fit([tr_ii, tr_am], tr_lab, epochs=epochs, batch_size=batch_size)
plot_learning_curve(history)

# score the baseline model
y_pred = model.predict([te_ii, te_am])
y_pred_bool = y_pred.round()
print(classification_report(te_lab, y_pred_bool))


# In[12]:


# build & train add feature 3-way model
learning_rate = 3e-5
epochs = 4
batch_size = 16
train_data_size = len(tr_lab)
model = build_model_propa_3way(model_base, learning_rate, epochs, batch_size, train_data_size)
history = model.fit([tr_ii, tr_am, tr_propa],
                   tr_lab, epochs=epochs, batch_size=batch_size)
plot_learning_curve(history)

# score the baseline model
y_pred = model.predict([te_ii, te_am, te_propa])
y_pred_bool = y_pred.round()
print(classification_report(te_lab, y_pred_bool))


# In[ ]:




