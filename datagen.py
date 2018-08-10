# __author__=   'Sargam Modak'

import json
import logging
import numpy as np
import pandas as pd
from keras.utils import to_categorical

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(name=__name__)


def generator(batch_size=32, mode='train'):
    """
    creates generator for training and validation dataset
    :param batch_size:
    :param mode: train or valid or test(not used)
    :return:
    """
    
    logger.info("creating generator for {} mode.".format(mode))
    
    if mode == 'train':
        train = dict(json.load(open('util_files/train.json')))
        ids = train['ids']
        data = train['data']
        
    elif mode == 'valid':
        valid = dict(json.load(open('util_files/valid.json')))
        ids = valid['ids']
        data = valid['data']
        
    elif mode == 'test':
        test = dict(json.load(open('util_files/test.json')))
        ids = test['ids']
        data = test['data']
        
    else:
        raise Exception("Not a valid mode. It should be either 'train', 'test' or 'valid'")
    
    df = pd.read_csv('data/document_departments.csv')
    labels = dict(df.values.tolist())
    
    cls2id = dict(json.load(open('util_files/cls2id.json')))
    num_classes = len(cls2id)
    
    cur_index = 0
    total = len(data)
    
    while True:
        start_index = cur_index
        end_index = min(cur_index+batch_size, total)
        x = data[start_index:end_index]
        y = ids[start_index:end_index]
        
        # converting label to its corresponding integral id
        for i in range(end_index-start_index):
            label = labels[int(y[i])]
            y[i] = cls2id[label]
        
        # one hot encoder
        y = to_categorical(np.asarray(y),
                           num_classes=num_classes)
        
        x = np.asarray(x, dtype=np.float32)
        
        # updating cur_index for next loop
        cur_index = end_index
        if end_index == total:
            cur_index = 0
        
        yield (x, y)
    
if __name__ == '__main__':
    gen = generator(batch_size=2)
    for _ in range(3):
        gen.next()
