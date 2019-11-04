
# coding: utf-8

# In[1]:


#! /bin/env python
# -*- coding: utf-8 -*-
"""
训练网络，并保存模型，其中LSTM的实现采用Python中的keras库
"""
import pandas as pd 
import numpy as np 
import jieba
import multiprocessing
import keras

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
from keras.callbacks import TensorBoard 
from keras.callbacks import LearningRateScheduler

np.random.seed(1337)  # For Reproducibility
import sys
sys.setrecursionlimit(1000000)
import yaml
import math

# set parameters:
cpu_count = multiprocessing.cpu_count() # 4核训练更快
vocab_dim = 100     # 词向量的维度
n_iterations = 1    # 训练词向量时，语料库上的迭代次数
n_exposures = 10    # 所有频数超过10的词语
window_size = 7     # 预测单词时所看的窗口大小
n_epoch = 70        # 训练网络的轮数
input_length = 100  # 网络中嵌入层的输入长度
maxlen = 100        # 句子的固定长度，不够maxlen就填充0

batch_size = 32     # 训练网络时的“步长”

#加载文件并整合，得到标签
def loadfile():
    neg=pd.read_csv('../data/neg.csv',header=None,index_col=None)
    pos=pd.read_csv('../data/pos.csv',header=None,index_col=None,error_bad_lines=False)
    neu=pd.read_csv('../data/mid.csv', header=None, index_col=None)

    combined = np.concatenate((pos[0], neu[0], neg[0]))
    y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neu), dtype=int), 
                        -1*np.ones(len(neg),dtype=int)))    # 标签

    return combined,y


#对句子经行分词，并去掉换行符
def tokenizer(text):
    ''' 
	    是一个简单的分词器，用jieba实现。
        用空格替换掉换行符。
    '''
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text

#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None, combined=None):
    ''' 这个函数做3件事
        1- 创建一个单词到索引的映射
        2- 创建一个单词到词向量的映射
        3- 对训练集和测试集的词典进行转换
    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        #  词频小于10->0 所以v->k+1
        w2indx = {v: k+1 for k, v in gensim_dict.items()} #所有频数超过10的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model[word] for word in w2indx.keys()} #所有频数超过10的词语的词向量, (word->model(word))

        def parse_dataset(combined): # 闭包-->临时使用, 把combined中的词语转换成对应的索引
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0) # 词频小于10->0
                data.append(new_txt)
            return data # word=>index
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen) #每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print ('No data provided...')


#用word2vec训练词向量，创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):

    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model.build_vocab(combined) # input: list
    model.train(combined, None, model.corpus_count,None,model.iter)
    model.save('../model/Word2vec_model.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined


def get_data(index_dict,word_vectors,combined,y):

    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim)) # 初始化 索引为0的词语，词向量全为0
    for word, index in index_dict.items(): # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    y_train = keras.utils.to_categorical(y_train,num_classes=3) # 变成二值类别矩阵
    y_test = keras.utils.to_categorical(y_test,num_classes=3)
    # print x_train.shape,y_train.shape
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test


#定义网络结构，采用Keras中提供的Sequential模型
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print( 'Defining a Simple Keras Model...')
    model = Sequential() 
	
	#加上嵌入层，将正整数（单词的索引们）转换为具有固定大小（词向量大小）的向量
    model.add(Embedding(output_dim=vocab_dim,  # 词向量维度
                        input_dim=n_symbols,   # 字典长度
                        mask_zero=True,        # '0'看作应被忽略的填充
                        weights=[embedding_weights],
                        input_length=input_length)) 
    model.add(LSTM(output_dim=50, activation='tanh')) # 循环层，用LSTM
    model.add(Dropout(0.5))                           # 防止过拟合，每次按0.5的比例断开神经元
    model.add(Dense(3, activation='softmax'))         # Dense=>全连接层,输出维度=3
    model.add(Activation('softmax'))                  # 激活层。激活函数softmax

    print ('Compiling the Model...')
    model.compile(loss='categorical_crossentropy',                      # 损失函数
                  optimizer='adam',metrics=['categorical_accuracy'])    # 优化方法， 以准确率为指标
    
    #tensorboard
    log_filepath = '../tmp/keras_log' 
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)  
    # 设置log的存储位置，将网络权值以图片格式保持在tensorboard中显示，设置每一个周期计算一次网络的权值，每层输出值的分布直方图 
    def step_decay(epoch):
        initial_lrate = 0.01
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
        return lrate
    lrate = LearningRateScheduler(step_decay)
    sgd = keras.optimizers.SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
    
    cbks = [tb_cb,lrate]  
    #history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=cbks, validation_data=(X_test, Y_test))

    
    
    print ("Train...") # batch_size=32；每次输出进度条记录
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch,verbose=1,callbacks=cbks,validation_data=(x_test, y_test))

    print ("Evaluate...")
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
	
    # 把网络结构用.yaml文件存起来，最后输出评估得分
    yaml_string = model.to_yaml()
    with open('../model/lstm.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('../model/lstm.h5')
    print ('Test score:', score)


#训练模型，并保存
print ('Loading Data...')
combined,y=loadfile()
print (len(combined),len(y))
print ('Tokenising...')
combined = tokenizer(combined)
print ('Training a Word2vec model...')
index_dict, word_vectors,combined=word2vec_train(combined)

print ('Setting up Arrays for Keras Embedding Layer...')
n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
print ("x_train.shape and y_train.shape:")
print (x_train.shape,y_train.shape)
train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)

