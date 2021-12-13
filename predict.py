import pandas as pd
import numpy as np

from joblib import load

from models.mlp import MLPClassifier

def predict(X, Y, encoder, model):
    y_pred = model.predict(X,encoder)


if __name__ == '__main__':
    # Setup
    X = pd.read_csv('checkpoints/x_test.csv') #load features
    Y = pd.read_csv('checkpoints/y_test.csv') #load labels
    encoder = load('checkpoints/encoder.joblib') #load encoder
    model = MLPClassifier.load(path='models', name='trained_model')

    y_pred = predict(X, Y, encoder, model)
