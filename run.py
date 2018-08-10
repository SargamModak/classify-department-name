import os
import json
import logging
import pandas as pd
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from test import predict
from datagen import generator
from create_model import get_model
from create_dataset import create_dataset

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(name=__name__)


def get_class_weights():
    """
    calculate class weights for training
    :return: dictionary key:class value:corresponding class weight
    """
    # labels from file id
    df = pd.read_csv('data/document_departments.csv')
    labels = dict(df.values.tolist())
    
    training_data = dict(json.load(open('util_files/train.json')))
    ids = training_data['ids']
    len_data = len(ids)
    
    cls2id = dict(json.load(open('util_files/cls2id.json')))
    
    class_count = {}
    
    # calculate count for each key
    for ind in ids:
        key = int(cls2id[labels[int(ind)]])
        if key in class_count:
            class_count[key] += 1
        else:
            class_count[key] = 1
            
    class_weights_dict = {}
    
    # calculate weight for each key
    for key in class_count.keys():
        class_weights_dict[key] = 1 - (class_count[key]*1./len_data)
        
    print "class weights"
    print class_weights_dict
    
    return class_weights_dict

utils_files_dir = 'util_files'
batch_size = 16

if not os.path.exists(utils_files_dir):
    os.mkdir(utils_files_dir)
    
if not os.path.exists(os.path.join(utils_files_dir, "train.json")):
    logger.info("creating and preparing data for training and testing.")
    create_dataset()

class_weights = get_class_weights()

if not os.path.exists(os.path.join(utils_files_dir, "model.h5")):
    logger.info("creating model.")
    model = get_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    train = dict(json.load(open('util_files/train.json')))
    train_data = train['data']
    len_training_data = len(train_data)

    valid = dict(json.load(open('util_files/valid.json')))
    valid_data = valid['data']
    len_validation_data = len(valid_data)

    training_generator = generator(batch_size=batch_size)
    validation_generator = generator(batch_size=batch_size,
                                     mode='valid')

    earlystopping = EarlyStopping(patience=5,
                                  verbose=1)

    model_checkpoint = ModelCheckpoint(filepath=os.path.join(utils_files_dir, "model.h5"),
                                       verbose=1,
                                       save_best_only=True)

    callbacks = [earlystopping, model_checkpoint]

    logger.info("training the model.")
    # train the model
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=len_training_data / batch_size,
                        epochs=100,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=validation_generator,
                        validation_steps=len_validation_data / batch_size,
                        class_weight=class_weights)
else:
    logger.info("loaded previously trained model successfully.")
    # load previous trained model
    model = load_model(filepath=os.path.join(utils_files_dir, "model.h5"))

logger.info("predicting and evaluating the testset.")
# prediction on test set
predict(model=model)
