import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pickle

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report

import joblib
from joblib import dump, load

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.callbacks.callbacks import EarlyStopping
from keras.utils import np_utils


import seaborn as sns
import matplotlib.pyplot as plt

class MLPClassifier:
    def __init__(self, n_class, input_shape) -> None:
        self.n_class = n_class
        self.input_shape = input_shape #(X[train_index].shape[1],)
        self.model = None

    def __str__(self) -> str:
        print(self.model.summary())


    def build(self):

        self.model = keras.models.Sequential([
                keras.layers.Dense(39,input_shape=self.input_shape,activation='relu'),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.6),
                keras.layers.Dense(self.n_class, activation='softmax')])

        self.model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])


    def fit(self, X, Y, batch_size=32, epochs=100, val_data=None, callbacks=False):

        early_stopping = keras.callbacks.EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)

        if self.model is not None:
            if callbacks==True:
                self.model.fit( x=X, y=Y,
                                batch_size=batch_size, epochs=epochs,
                                validation_data=val_data,
                                callbacks=[early_stopping])
            else:
                self.model.fit( x=X, y=Y,
                                batch_size=batch_size, epochs=epochs,
                                validation_data=val_data)
        else:
            raise RuntimeError('The model is not defined. You have to create one before fitting it')


    def eval(self, X, Y):
        scores = self.model.evaluate(x=X, y=Y, verbose=0)
        return scores


    def predict(self, x_test, encoder):

        y = self.model.predict(x_test)
        y_pred = encoder.inverse_transform(y)

        return y_pred


    def confusion_matrix(self, x_test, y_test, encoder, save: bool=True):
        
        y_pred = self.predict(self, x_test, encoder)
        cm = confusion_matrix(y_test, y_pred, normalize='pred')
        plt.figure(figsize = (12, 10))
        cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
        sns.heatmap(cm, linecolor='white', cmap='rocket_r', linewidth=1, annot=True, fmt='.1%')
        plt.title('Confusion Matrix', size=20)
        plt.xlabel('Predicted Labels', size=14)
        plt.ylabel('Actual Labels', size=14)
        plt.show()
        if save:
            plt.savefig('out/figures/confusion_matrix.png')


    def classification_report(self, x_test, y_test, encoder, save: bool=True):
        y_pred = self.predict(self, x_test, encoder)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        if save:
            df.to_csv('out/classification_report.csv', index = False)
        return df


    def save(self, path: str, name: str) -> None:
        """
        Args:
            path (str):
            name (str):
        """
        save_path = os.path.abspath(os.path.join(path, name + '.m'))
        pickle.dump(self, open(save_path, "wb"))

    @classmethod
    def load(cls, path: str, name: str):
        '''
        Args:
            path (str): 
            name (str): 
        '''
        model_path = os.path.abspath(os.path.join(path, name + '.m'))
        with open(model_path, 'rb') as pickle_file:
            model = pickle.load(pickle_file)
        return model