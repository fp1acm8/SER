import pandas as pd
import numpy as np
from numpy import loadtxt


from joblib import load

from models.mlp import MLPClassifier

def predict(X, Y, encoder, model):

    model.predict(X,encoder)
    model.confusion_matrix(X, Y, encoder)
    model.classification_report(X, Y, encoder)


if __name__ == '__main__':
    # Setup
    X = loadtxt('checkpoints/x_test.csv', delimiter=',') #load features
    Y = loadtxt('checkpoints/y_test.csv', delimiter=',') #load labels
    encoder = load('checkpoints/encoder.joblib') #load encoder
    model = MLPClassifier.load(path='models', name='trained_model') #load trained model

    predict(X, Y, encoder, model)
