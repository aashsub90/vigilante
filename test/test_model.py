
# coding: utf-8

# # Test Model

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report
import sys

def loadModel(name):
    '''
    function to load model object with the given name.
    '''
    with open('randomForest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = loadModel('randomForest')
with open('sample.pkl', 'rb') as f:
        X_test = pickle.load(f)
with open('actual.pkl', 'rb') as f:
        y_test = pickle.load(f)
y_pred = model.predict(X_test)
# Predicting the classes of test data
with open('prediction.pkl','wb') as f:
    pickle.dump(y_pred, f)
print(classification_report(y_test, y_pred, target_names=['high', 'low', 'moderate']))