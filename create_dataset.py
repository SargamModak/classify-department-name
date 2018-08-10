# __author__=   'Sargam Modak'

import os
import re
import json
import random
import pickle
import logging
import numpy as np
import pandas as pd
from math import ceil
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


MAX_NUM_WORDS = 5000
MAX_SEQUENCE_LENGTH = 1000
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(name=__name__)


def get_data(path):
    """
    create input data from files
    :param path: path to folder containing json files
    :return: dictionary key: class, value: list of all data for that class
    """
    assert os.path.exists(path), "{} path is not correct.".format(path)
    
    files = os.listdir(path)
    data_dict = {}

    df = pd.read_csv('data/document_departments.csv')
    labels = dict(df.values.tolist())
    
    for cur_file in files:
        
        with open(os.path.join('data', 'docs', cur_file)) as f:
            data = json.load(f)
        
        # job description
        job_desc = data['jd_information']['description']
        
        # replace '&nbsp;' with ' ' in job description
        job_desc = re.sub(r"&[a-z]*;", " ", job_desc)
        
        # job keywords
        job_keywords = data['api_data']['job_keywords']
        
        words = job_desc.split(" ")
        
        new_job_desc = ""
        
        for word in words:
            flag = False

            # converting words like 'fromOffice' to 'from office' in the data
            for ind, letter in enumerate(word[1:]):
                if letter.isupper():
                    flag = True
                    break
            
            if flag and ind > 0:
                new_job_desc += word[:ind + 1] + " " + word[ind + 1:] + " "
            else:
                new_job_desc += word + " "
        
        for keyword in job_keywords:
            new_job_desc += " " + keyword
        
        # if neither job description nor job keywords are present
        if len(new_job_desc) < 1:
            logger.info("{} file has no job description and job keywords.".format(data['_id']))
            continue
        
        key = labels[int(data['_id'])]
        if key in data_dict:
            data_dict[key].append([data['_id'], new_job_desc])
        else:
            data_dict[key] = [[data['_id'], new_job_desc]]
        
    return data_dict


def get_tokenizer(data):
    """
    get tokenizer based on training data
    :param data: training data
    :return: tokenizer
    """
    
    texts = [d[1] for d in data]
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts=texts)
    
    logger.info(msg='{} number of words in tokeninzer'.format(len(tokenizer.word_index)))
    
    return tokenizer


def get_tokenized(tokenizer, data):
    """
    create tokens for the data passed
    :param tokenizer: tokenizer object
    :param data: data to be tokenized
    :return: tokenized and padded to a fixed length data
    """
    texts = [d[1] for d in data]
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences=sequences,
                                     maxlen=MAX_SEQUENCE_LENGTH)
    return padded_sequences


def create_dataset(train_per=0.8, valid_per=0.1):
    """
    collect data from files and create training, validation and testing data
    :param train_per: training percentage
    :param valid_per: percentage of training data to be used for validation
    :return: tokenized training, validation and testing data
    """
    logger.info("creating dataset.")
    data_dict = get_data('data/docs')
    
    data_train = []
    data_valid = []
    data_test = []
    
    for key in data_dict.keys():
        data = data_dict[key]
        total_len = len(data)
        
        if total_len < 5:
            logger.info("{} is not included in training and testing as it less than 5 instances overall.".format(key))
            continue
            
        train_valid_len = int(ceil(total_len * train_per))
        
        if train_valid_len < 2 or train_valid_len > 9:
            valid_len = int(train_valid_len * valid_per)
            train_len = train_valid_len - valid_len
            
        else:
            valid_len = 1
            train_len = train_valid_len - valid_len
        
        data_train_valid = data[:train_valid_len]
        
        data_train.extend(data_train_valid[:train_len])
        data_valid.extend(data_train_valid[train_len:])
        
        data_test.extend(data[train_valid_len:])
        
    logger.info("shuffling training and testing dataset.")
    random.shuffle(x=data_train)
    random.shuffle(x=data_valid)
    random.shuffle(x=data_test)
    
    logger.info("creating tokenizer from training dataset.")
    tokenizer = get_tokenizer(data_train)
    
    pickle.dump(obj=tokenizer,
                file=open(name='util_files/tokenizer',
                          mode='w'))
    
    logger.info("tokenizing training dataset.")
    tokenized_data_train = get_tokenized(tokenizer, data_train)

    logger.info("tokenizing validation dataset.")
    tokenized_data_valid = get_tokenized(tokenizer, data_valid)

    logger.info("tokenizing testing dataset.")
    tokenized_data_test = get_tokenized(tokenizer, data_test)
    
    logger.info("saving tokenized training dataset.")
    with open('util_files/train.json', 'w') as f:
        ids = [data[0] for data in data_train]
        d = {'ids': ids, 'data': tokenized_data_train.tolist()}
        json.dump(d, f)

    logger.info("saving tokenized validation dataset.")
    with open('util_files/valid.json', 'w') as f:
        ids = [data[0] for data in data_valid]
        d = {'ids': ids, 'data': tokenized_data_valid.tolist()}
        json.dump(d, f)

    logger.info("saving tokenized testing dataset.")
    with open('util_files/test.json', 'w') as f:
        ids = [data[0] for data in data_test]
        d = {'ids': ids, 'data': tokenized_data_test.tolist()}
        json.dump(d, f)
        
    df = pd.read_csv('data/document_departments.csv')
    df_list = df.values.tolist()
    classes = np.unique([x[1] for x in df_list])
    
    cls2id = {}
    id2cls = {}
    
    for idx, cls in enumerate(classes):
        cls2id[cls] = idx
        id2cls[idx] = cls

    logger.info("saving dictionary of id to corresponding class")
    with open('util_files/id2cls.json', 'w') as f:
        json.dump(id2cls, f)

    logger.info("saving dictionary of class to corresponding id")
    with open('util_files/cls2id.json', 'w') as f:
        json.dump(cls2id, f)


if __name__ == '__main__':
    if not os.path.exists("util_files"):
        os.mkdir("util_files")
    create_dataset()
