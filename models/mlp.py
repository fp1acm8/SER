import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pickle

from sklearn.metrics import confusion_matrix, classification_report

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
        ''' Define model architecture using Keras 
        '''
        self.model = keras.models.Sequential([
                keras.layers.Dense(39,input_shape=self.input_shape,activation='relu'),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.6),
                keras.layers.Dense(self.n_class, activation='softmax')])

        self.model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])


    def fit(self, X, Y, batch_size=32, epochs=100, val_data=None, callbacks=False):
        ''' Trains the model for a fixed number of epochs (iterations on a dataset).
            -----------------------------------------------------------------------
            Args:
            > X: Input data
            > Y: Target data
            > batch_size: Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32.
            > epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. If unspecified, epochs will default to 100. 
            > val_data: Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. A tuple (x_val, y_val) of Numpy arrays or tensors.
            > callbacks: If True, early stopping [EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)] is applied during training.
        '''
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
         ''' Returns the loss value & metrics values for the model in test mode.
            -----------------------------------------------------------------------
            Args:
            > X: Input data
            > Y: Target data
        '''
        scores = self.model.evaluate(x=X, y=Y, verbose=0)
        return scores


    def predict(self, X, encoder, save: bool=True):
          ''' Generates output predictions for the input samples.
            -----------------------------------------------------------------------
            Args:
            > X: Input samples
            > encoder: object used to encode categorical features
            > save: if True the predictions are saved as 'out/predictions.csv'
        '''
        y = self.model.predict(X)
        y_pred = encoder.inverse_transform(y)
        if save: 
            y_pred_df = pd.DataFrame(y_pred)
            y_pred_df.to_csv('out/predictions.csv', index=False)
        
        return y_pred


    def confusion_matrix(self, X, Y, encoder, save: bool=True):
        ''' Compute confusion matrix to evaluate the accuracy of a classification.
            -----------------------------------------------------------------------
            Args:
            > X: Input samples
            > Y: Ground truth (correct) target values.
            > encoder: object used to encode categorical features
            > save: if True the figure is saved as 'out/confusion_matrix.png'
        '''
        y_pred = self.predict(X, encoder)
        Y = encoder.inverse_transform(Y)
        cm = confusion_matrix(Y, y_pred, normalize='pred')
        plt.figure(figsize = (12, 10))
        cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
        sns.heatmap(cm, linecolor='white', cmap='rocket_r', linewidth=1, annot=True, fmt='.1%')
        plt.title('Confusion Matrix', size=20)
        plt.xlabel('Predicted Labels', size=14)
        plt.ylabel('Actual Labels', size=14)
        plt.show()
        if save==True:
            plt.savefig('out/confusion_matrix.png')


    def classification_report(self, X, Y, encoder, save: bool=True):
        ''' Build a report showing the main classification metrics.
            -----------------------------------------------------------------------
            Args:
            > X: Input samples
            > Y: Ground truth (correct) target values.
            > encoder: object used to encode categorical features
            > save: if True the report is saved as 'out/classification_report.csv'
        '''
        y_pred = self.predict(X, encoder)
        Y = encoder.inverse_transform(Y)
        report = classification_report(Y, y_pred, output_dict=True)
        if save:
            df = pd.DataFrame(report).transpose()
            df.to_csv('out/classification_report.csv', index=True)


    def save(self, path: str, name: str) -> None:
        '''
        Args:
            path (str): path
            name (str): file name
        '''
        save_path = os.path.abspath(os.path.join(path, name + '.m'))
        pickle.dump(self, open(save_path, "wb"))


    @classmethod
    def load(cls, path: str, name: str):
        '''
        Args:
            path (str): path
            name (str): file name
        '''
        model_path = os.path.abspath(os.path.join(path, name + '.m'))
        with open(model_path, 'rb') as pickle_file:
            model = pickle.load(pickle_file)
        return model
