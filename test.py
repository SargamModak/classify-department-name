# __author__=   'Sargam Modak'

import json
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score


def predict(model):
    """
    prediction on testing dataset and calculating accuracy for test set
    :param model: trained model
    :return:
    """
    # load test dataset
    test = dict(json.load(open('util_files/test.json')))
    ids = test['ids']
    data = test['data']

    df = pd.read_csv('data/document_departments.csv')
    labels = dict(df.values.tolist())

    id2cls = dict(json.load(open('util_files/id2cls.json')))
    
    ytrue = []
    ypredicted = []
    
    for i in range(len(data)):
        
        prediction = np.argmax(model.predict_on_batch(np.expand_dims(data[i], axis=0)))
        
        ypredicted.append(id2cls[str(prediction)])
        
        cls = labels[int(ids[i])]
        ytrue.append(cls)
    
    print "classification report"
    print classification_report(y_true=ytrue,
                                y_pred=ypredicted)
    
    print "*********************"
    print "Accuracy on test set"
    print accuracy_score(y_true=ytrue,
                         y_pred=ypredicted)
    print "*********************"
    
if __name__ == '__main__':
    trained_model = load_model("util_files/model.h5")
    predict(model=trained_model)
